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
from routers.realtime_router import _kafka_inference_listener
from patient_state_machine import PatientSimulator, PatientState
from kafka_config import get_kafka_config, is_cloud_kafka
from config import settings

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

# Pull intervals from config (single source of truth); env var overrides still work
TICK_INTERVAL = int(os.getenv("TICK_INTERVAL", str(settings.TICK_INTERVAL)))
PATIENT_POLL_INTERVAL = int(os.getenv("PATIENT_POLL_INTERVAL", str(settings.PATIENT_POLL_INTERVAL)))

# ─── Self base URL — always resolved at call time, never at import time ───────
def _get_self_base() -> str:
    explicit = os.getenv("SELF_BASE")
    if explicit:
        return explicit
    port = os.getenv("PORT", "10000")
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
        timeout=10,
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
            log.error("[Simulator] Error in sim loop for %s: %s", patient_id, e)

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
# Vitals Consumer — uses batch risk endpoint
# ═══════════════════════════════════════════════════════════════════════════════

async def _vitals_consumer_loop():
    """
    Consumes vitals.raw, POSTs to /icu/vitals, gets AI risk via batch endpoint,
    publishes to inference.output.
    """
    kafka_cfg = get_kafka_config()
    await asyncio.sleep(5)
    log.info("[VitalsConsumer] Starting...")

    # Accumulate patient IDs seen in one flush window before batching risk call
    _pending_risk: Dict[str, dict] = {}  # {patient_id: {vitals, api_response, simulator_state}}

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

                    # ── Step 1: POST vitals ────────────────────────────────────
                    try:
                        token = await _get_token(client)
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

                        # Queue patient for batch risk calculation
                        _pending_risk[patient_id] = {
                            "vitals": vitals,
                            "api_response": api_response,
                            "simulator_state": simulator_state,
                        }

                    except httpx.TimeoutException:
                        log.warning("[VitalsConsumer] Vitals POST timed out for %s — skipping", patient_id)
                        continue
                    except httpx.HTTPStatusError as e:
                        log.error("[VitalsConsumer] API error for %s: %s %s",
                                  patient_id, e.response.status_code, e.response.text[:200])
                        continue
                    except Exception as e:
                        log.error("[VitalsConsumer] Unexpected error ingesting vitals for %s: %s", patient_id, e)
                        continue

                    # ── Step 2: Batch risk for all pending patients ────────────
                    # Drain all currently buffered patient IDs in one HTTP call.
                    if not _pending_risk:
                        continue

                    pending_ids = list(_pending_risk.keys())
                    batch_risks: Dict[str, dict] = {}

                    try:
                        token = await _get_token(client)
                        resp2 = await client.post(
                            f"{base}/icu/ai/risk/batch",
                            json={"patient_ids": pending_ids},
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=15,
                        )
                        resp2.raise_for_status()
                        batch_result = resp2.json()
                        batch_risks = batch_result.get("results", {})
                        cached = batch_result.get("cached_count", 0)
                        log.info("[VitalsConsumer] Batch risk: %d patients | %d cached",
                                 len(pending_ids), cached)

                    except httpx.TimeoutException:
                        log.warning("[VitalsConsumer] Batch risk timed out — skipping risk for %s",
                                    pending_ids)
                        _pending_risk.clear()
                        continue
                    except httpx.HTTPStatusError as e:
                        log.error("[VitalsConsumer] Batch risk API error: %s %s",
                                  e.response.status_code, e.response.text[:200])
                        _pending_risk.clear()
                        continue
                    except Exception as e:
                        log.error("[VitalsConsumer] Batch risk unexpected error: %s", e)
                        _pending_risk.clear()
                        continue

                    # ── Step 3: Publish each result to inference.output ────────
                    for pid, pending in _pending_risk.items():
                        risk = batch_risks.get(pid, {})
                        if "error" in risk:
                            log.warning("[VitalsConsumer] Risk error for %s: %s", pid, risk["error"])
                            continue

                        output = {
                            "patient_id": pid,
                            "vitals": pending["vitals"],
                            "vitals_response": pending["api_response"],
                            "risk": risk,
                            "simulator_state": pending["simulator_state"],
                            "timestamp": time.time(),
                        }
                        try:
                            await producer.send_and_wait(
                                "inference.output",
                                key=pid.encode(),
                                value=json.dumps(output).encode(),
                            )
                        except Exception as e:
                            log.error("[VitalsConsumer] Failed to publish inference output for %s: %s", pid, e)

                    _pending_risk.clear()

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
                    except httpx.TimeoutException:
                        log.warning("[LabsConsumer] Labs POST timed out for %s — skipping", patient_id)
                    except httpx.HTTPStatusError as e:
                        log.error("[LabsConsumer] Labs API error for %s: %s — %s",
                                  patient_id, e.response.status_code, e.response.text[:300])
                    except Exception as e:
                        log.error("[LabsConsumer] Error posting labs for %s: %s — type=%s",
                                  patient_id, e, type(e).__name__, exc_info=True)

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
# WebSocket Broadcaster
# ═══════════════════════════════════════════════════════════════════════════════

async def _inference_ws_broadcaster():
    """Consume inference.output → broadcast to WebSocket subscribers."""
    from routers.realtime_router import manager  # import here to avoid circular
    kafka_cfg = get_kafka_config()
    await asyncio.sleep(8)

    while True:
        consumer = None
        try:
            consumer = AIOKafkaConsumer(
                "inference.output",
                **kafka_cfg,
                group_id="ws-broadcast-group",
                auto_offset_reset="latest",
                enable_auto_commit=True,
                value_deserializer=lambda v: json.loads(v.decode()),
            )
            await consumer.start()
            log.info("[WSBroadcaster] Connected to inference.output")

            async for msg in consumer:
                data = msg.value
                patient_id = data["patient_id"]
                risk_obj = data.get("risk", {})
                ra = risk_obj.get("risk_assessment", {})

                payload = {
                    "type": "inference_update",
                    "patient_id": patient_id,
                    "riskPercentage": ra.get("overall_score", 0),
                    "label": ra.get("category", "LOW RISK"),
                    "mort_7d": ra.get("mort_7d"),
                    "mort_30d": ra.get("mort_30d"),
                    "sofa_score": ra.get("sofa_score"),
                    "factors": risk_obj.get("contributing_factors", []),
                    "vitals": data.get("vitals", {}),
                    "simulator_state": data.get("simulator_state", "stable"),
                    "timestamp": data.get("timestamp"),
                }
                try:
                    await manager.broadcast(patient_id, payload)
                except Exception as e:
                    log.error("[WSBroadcaster] Broadcast error for %s: %s", patient_id, e)

        except asyncio.CancelledError:
            log.info("[WSBroadcaster] Shutting down")
            break
        except Exception as e:
            log.error("[WSBroadcaster] %s — retrying in 5s", e)
            await asyncio.sleep(5)
        finally:
            if consumer:
                try:
                    await consumer.stop()
                except Exception:
                    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def start_pipeline():
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
    _tasks.append(asyncio.create_task(_inference_ws_broadcaster()))
    _tasks.append(asyncio.create_task(_kafka_inference_listener()))


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
