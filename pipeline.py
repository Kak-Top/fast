"""
Embedded Pipeline — Runs Simulator + Consumers inside FastAPI
=============================================================
This replaces the 3 terminal windows (simulator, vitals consumer, labs consumer)
with background asyncio tasks that run inside your FastAPI process.

All Kafka communication is REAL Kafka (Aiven cloud or local).
Consumers POST to your existing FastAPI endpoints via HTTP (localhost).

HOW TO ADD to your main.py:
    from pipeline import start_pipeline, stop_pipeline

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await start_pipeline()
        yield
        await stop_pipeline()
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict

import httpx
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from patient_state_machine import PatientSimulator, PatientState
from kafka_config import get_kafka_config, is_cloud_kafka

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ─── Config ───────────────────────────────────────────────────────────────────
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "admin123")
VITALS_TOPIC = "vitals.raw"
LABS_TOPIC = "labs.results"
TICK_INTERVAL = int(os.getenv("TICK_INTERVAL", "10"))
PATIENT_POLL_INTERVAL = int(os.getenv("PATIENT_POLL_INTERVAL", "30"))

# ─── Self base URL — always resolved at call time, never at import time ───────
def _get_self_base() -> str:
    # SELF_BASE env var takes priority (set this in Render to http://127.0.0.1:10000)
    # Falls back to PORT env var, then hardcoded 10000 (Render always uses 10000)
    explicit = os.getenv("SELF_BASE")
    if explicit:
        return explicit
    port = os.getenv("PORT", "10000")
    # Guard against Render passing the literal string "${PORT}" before substitution
    if not port.isdigit():
        port = "10000"
    return f"http://127.0.0.1:{port}"

# ─── State ────────────────────────────────────────────────────────────────────
_tasks = []
_sim_tasks: Dict[str, asyncio.Task] = {}
_simulators: Dict[str, PatientSimulator] = {}
_producer: AIOKafkaProducer = None
_token: str = ""
_token_issued_at: float = 0.0
TOKEN_TTL = 55 * 60


# ═══════════════════════════════════════════════════════════════════════════════
# Auth helpers
# ═══════════════════════════════════════════════════════════════════════════════

async def _get_token(client: httpx.AsyncClient) -> str:
    """Authenticate with FastAPI and cache the token."""
    global _token, _token_issued_at
    if _token and (time.time() - _token_issued_at) < TOKEN_TTL:
        return _token
    base = _get_self_base()
    resp = await client.post(
        f"{base}/auth/login",
        data={"username": API_USERNAME, "password": API_PASSWORD},
    )
    resp.raise_for_status()
    _token = resp.json()["access_token"]
    _token_issued_at = time.time()
    log.info("Authenticated as %s", API_USERNAME)
    return _token


async def _fetch_patients(client: httpx.AsyncClient) -> list:
    """Fetch current patients from the API."""
    token = await _get_token(client)
    base = _get_self_base()
    resp = await client.get(
        f"{base}/icu/patients",
        headers={"Authorization": f"Bearer {token}"},
    )
    resp.raise_for_status()
    return resp.json().get("patients", [])


def _create_simulator(p: dict) -> PatientSimulator:
    """Create a PatientSimulator from API patient data."""
    sim = PatientSimulator(
        patient_id=p["patient_id"],
        name=p.get("name", "Unknown"),
        age=p.get("age", 60),
        diagnosis=p.get("diagnosis", "default"),
    )
    status = p.get("status", "stable")
    if status == "critical":
        sim.state = PatientState.CRITICAL
    elif status == "deteriorating":
        sim.state = PatientState.DETERIORATING
    else:
        sim.state = PatientState.STABLE
    return sim


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator — publishes to Kafka
# ═══════════════════════════════════════════════════════════════════════════════

async def _single_patient_sim_loop(patient_id: str):
    """Simulation loop for ONE patient. Runs until cancelled."""
    global _producer
    sim = _simulators[patient_id]
    log.info("[Simulator] Starting loop for %s (%s) — state=%s",
             patient_id, sim.name, sim.state.value)

    while True:
        try:
            # ── Vitals ──
            vitals = sim.tick_vitals()
            meta = vitals.pop("_meta")
            message = {
                "patient_id": patient_id,
                "vitals": vitals,
                "state": meta["state"],
                "timestamp": time.time(),
            }
            await _producer.send_and_wait(
                VITALS_TOPIC,
                key=patient_id.encode(),
                value=json.dumps(message).encode(),
            )
            log.info(
                "→ vitals.raw | %s | state=%-13s | HR=%.0f sys=%.0f spo2=%.0f",
                patient_id, meta["state"],
                vitals["heart_rate"], vitals["blood_pressure_sys"], vitals["spo2"],
            )

            # ── Labs ──
            labs = sim.tick_labs()
            if labs is not None:
                labs.pop("_meta")
                lab_msg = {
                    "patient_id": patient_id,
                    "labs": labs,
                    "timestamp": time.time(),
                }
                await _producer.send_and_wait(
                    LABS_TOPIC,
                    key=patient_id.encode(),
                    value=json.dumps(lab_msg).encode(),
                )
                log.info("→ labs.results | %s | glucose=%.0f",
                         patient_id, labs["glucose"])

        except asyncio.CancelledError:
            log.info("[Simulator] Stopped loop for %s", patient_id)
            break
        except Exception as e:
            log.error("[Simulator] Error for %s: %s", patient_id, e)

        await asyncio.sleep(TICK_INTERVAL)


async def _patient_discovery_loop():
    global _producer
    kafka_cfg = get_kafka_config()

    await asyncio.sleep(3)
    log.info("[Simulator] Starting patient discovery...")
    log.info("[Simulator] Self base URL: %s", _get_self_base())

    _producer = AIOKafkaProducer(
        **kafka_cfg,
        compression_type="gzip",
        acks="all",
    )

    # Connect producer with timeout + retry
    while True:
        try:
            await asyncio.wait_for(_producer.start(), timeout=15.0)
            log.info("[Simulator] Kafka producer connected")
            break
        except asyncio.TimeoutError:
            log.error("[Simulator] Kafka producer timed out — retrying in 5s")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            return
        except Exception as e:
            log.error("[Simulator] Cannot connect Kafka producer: %s — retrying in 5s", e)
            await asyncio.sleep(5)

    # Discovery loop — runs forever, checks for new patients
    async with httpx.AsyncClient(timeout=15) as client:
        while True:
            try:
                patients = await _fetch_patients(client)
                log.info("[Simulator] Polled API — found %d patients", len(patients))

                for p in patients:
                    pid = p["patient_id"]
                    if pid in _sim_tasks and not _sim_tasks[pid].done():
                        continue
                    sim = _create_simulator(p)
                    _simulators[pid] = sim
                    task = asyncio.create_task(_single_patient_sim_loop(pid))
                    _sim_tasks[pid] = task
                    log.info(
                        "[Simulator] ★ NEW patient detected: %s (%s) — auto-started simulation",
                        pid, sim.name,
                    )

            except asyncio.CancelledError:
                log.info("[Simulator] Patient discovery shutting down")
                break
            except Exception as e:
                log.warning("[Simulator] Patient poll failed: %s — will retry", e)

            await asyncio.sleep(PATIENT_POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════════
# Vitals Consumer
# ═══════════════════════════════════════════════════════════════════════════════

async def _vitals_consumer_loop():
    """Consumes vitals.raw, POSTs to /icu/vitals, gets AI risk, publishes to inference.output."""
    kafka_cfg = get_kafka_config()
    await asyncio.sleep(5)
    log.info("[VitalsConsumer] Starting...")

    while True:
        consumer = None
        producer = None
        try:
            consumer = AIOKafkaConsumer(
                VITALS_TOPIC,
                **kafka_cfg,
                group_id="vitals-ingestion-group",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode()),
            )
            producer = AIOKafkaProducer(**kafka_cfg, compression_type="gzip")

            await consumer.start()
            await producer.start()
            log.info("[VitalsConsumer] Connected to Kafka — consuming vitals.raw")

            async with httpx.AsyncClient(timeout=15) as client:
                async for msg in consumer:
                    data = msg.value
                    patient_id = data["patient_id"]
                    vitals = data["vitals"]
                    simulator_state = data.get("state", "unknown")
                    base = _get_self_base()

                    try:
                        token = await _get_token(client)

                        # 1. POST vitals
                        resp = await client.post(
                            f"{base}/icu/vitals/{patient_id}",
                            json=vitals,
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        api_response = resp.json()
                        flags = api_response.get("abnormal_flags", [])
                        is_critical = api_response.get("is_critical", False)
                        log.info("✓ %s vitals ingested | critical=%s | flags=%d",
                                 patient_id, is_critical, len(flags))

                        # 2. Get AI risk
                        resp2 = await client.get(
                            f"{base}/icu/ai/risk/{patient_id}",
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10,
                        )
                        resp2.raise_for_status()
                        risk = resp2.json()

                        # 3. Publish to inference.output
                        output = {
                            "patient_id": patient_id,
                            "vitals": vitals,
                            "vitals_response": api_response,
                            "risk": risk,
                            "simulator_state": simulator_state,
                            "timestamp": time.time(),
                        }
                        await producer.send_and_wait(
                            "inference.output",
                            key=patient_id.encode(),
                            value=json.dumps(output).encode(),
                        )

                    except httpx.HTTPStatusError as e:
                        log.error("API error for %s: %s %s",
                                  patient_id, e.response.status_code, e.response.text[:200])
                    except Exception as e:
                        log.error("Error processing vitals for %s: %s", patient_id, e)

        except asyncio.CancelledError:
            log.info("[VitalsConsumer] Shutting down")
            break
        except Exception as e:
            log.error("[VitalsConsumer] Error: %s — retrying in 5s", e)
            await asyncio.sleep(5)
        finally:
            try:
                if consumer:
                    await consumer.stop()
                if producer:
                    await producer.stop()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Labs Consumer
# ═══════════════════════════════════════════════════════════════════════════════

async def _labs_consumer_loop():
    """Consumes labs.results, POSTs to /icu/labs/{patient_id}."""
    kafka_cfg = get_kafka_config()
    await asyncio.sleep(5)
    log.info("[LabsConsumer] Starting...")

    while True:
        consumer = None
        try:
            consumer = AIOKafkaConsumer(
                LABS_TOPIC,
                **kafka_cfg,
                group_id="labs-ingestion-group",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode()),
            )
            await consumer.start()
            log.info("[LabsConsumer] Connected to Kafka — consuming labs.results")

            async with httpx.AsyncClient(timeout=15) as client:
                async for msg in consumer:
                    data = msg.value
                    patient_id = data["patient_id"]
                    labs = data["labs"]
                    base = _get_self_base()

                    try:
                        token = await _get_token(client)
                        resp = await client.post(
                            f"{base}/icu/labs/{patient_id}",
                            json=labs,
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10,
                        )
                        resp.raise_for_status()
                        result = resp.json()
                        log.info(
                            "✓ %s labs ingested | glucose=%.0f creatinine=%.2f wbc=%.1f lactate=%.1f | flags=%d",
                            patient_id, labs["glucose"], labs["creatinine"],
                            labs["wbc"], labs["lactate"],
                            len(result.get("abnormal_flags", [])),
                        )
                    except httpx.HTTPStatusError as e:
                        log.error("API error for %s: %s %s",
                                  patient_id, e.response.status_code, e.response.text[:200])
                    except Exception as e:
                        log.error("Error posting labs for %s: %s", patient_id, e)

        except asyncio.CancelledError:
            log.info("[LabsConsumer] Shutting down")
            break
        except Exception as e:
            log.error("[LabsConsumer] Error: %s — retrying in 5s", e)
            await asyncio.sleep(5)
        finally:
            try:
                if consumer:
                    await consumer.stop()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def start_pipeline():
    """Start all pipeline tasks. Call from FastAPI lifespan."""
    log.info("═" * 50)
    log.info("Starting embedded Kafka pipeline...")
    log.info("  Kafka: %s", os.getenv("KAFKA_BOOTSTRAP", "localhost:9092"))
    log.info("  Cloud Kafka: %s", "YES (SSL)" if is_cloud_kafka() else "NO (local)")
    log.info("  Self API: %s", _get_self_base())
    log.info("  Tick interval: %ds", TICK_INTERVAL)
    log.info("  Patient poll interval: %ds", PATIENT_POLL_INTERVAL)
    log.info("═" * 50)

    _tasks.append(asyncio.create_task(_patient_discovery_loop()))
    _tasks.append(asyncio.create_task(_vitals_consumer_loop()))
    _tasks.append(asyncio.create_task(_labs_consumer_loop()))


async def stop_pipeline():
    """Stop all pipeline tasks. Call from FastAPI lifespan."""
    global _producer
    log.info("Stopping pipeline tasks...")

    for pid, task in _sim_tasks.items():
        task.cancel()
    if _sim_tasks:
        await asyncio.gather(*_sim_tasks.values(), return_exceptions=True)
    _sim_tasks.clear()
    _simulators.clear()

    for task in _tasks:
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    _tasks.clear()

    if _producer:
        try:
            await _producer.stop()
        except Exception:
            pass
        _producer = None

    log.info("Pipeline stopped.")
