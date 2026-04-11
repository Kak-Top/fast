"""
Embedded Pipeline — Runs Simulator + Consumers inside FastAPI
=============================================================
This replaces the 3 terminal windows (simulator, vitals consumer, labs consumer)
with background asyncio tasks that run inside your FastAPI process.

All Kafka communication is REAL Kafka (Upstash cloud or local).
Consumers POST to your existing FastAPI endpoints via HTTP (localhost).

HOW TO ADD to your main.py:
    from pipeline import start_pipeline, stop_pipeline

    @app.on_event("startup")
    async def startup():
        await start_pipeline()

    @app.on_event("shutdown")
    async def shutdown():
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
from kafka_config import get_kafka_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
API_USERNAME = os.getenv("API_USERNAME", "dr.ahmad")
API_PASSWORD = os.getenv("API_PASSWORD", "password123")
VITALS_TOPIC = "vitals.raw"
LABS_TOPIC = "labs.results"
TICK_INTERVAL = int(os.getenv("TICK_INTERVAL", "10"))

# When running embedded, use localhost to POST to self
SELF_BASE = os.getenv("SELF_BASE", "http://127.0.0.1:" + os.getenv("PORT", "8000"))

# ─── How often to poll the API for new patients (seconds) ────────────────────
PATIENT_POLL_INTERVAL = int(os.getenv("PATIENT_POLL_INTERVAL", "30"))

# ─── State ────────────────────────────────────────────────────────────────────
_tasks = []
_sim_tasks: Dict[str, asyncio.Task] = {}  # patient_id -> running simulation task
_simulators: Dict[str, PatientSimulator] = {}  # patient_id -> simulator instance
_producer: AIOKafkaProducer = None  # shared Kafka producer
_token: str = ""
_token_issued_at: float = 0.0
TOKEN_TTL = 55 * 60  # refresh every 55 min


async def _get_token(client: httpx.AsyncClient) -> str:
    """Authenticate with FastAPI and cache the token."""
    global _token, _token_issued_at
    if _token and (time.time() - _token_issued_at) < TOKEN_TTL:
        return _token
    resp = await client.post(
        f"{SELF_BASE}/auth/login",
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
    resp = await client.get(
        f"{SELF_BASE}/icu/patients",
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
# Simulator — publishes to Kafka (same as simulator_runner.py)
# ═══════════════════════════════════════════════════════════════════════════════

async def _single_patient_sim_loop(patient_id: str):
    """
    Simulation loop for ONE patient. Publishes vitals every TICK_INTERVAL
    and labs on schedule to Kafka. Runs until cancelled.
    """
    global _producer
    sim = _simulators[patient_id]
    log.info("[Simulator] Starting loop for %s (%s) — state=%s", patient_id, sim.name, sim.state.value)

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
                log.info("→ labs.results | %s | glucose=%.0f", patient_id, labs["glucose"])

        except asyncio.CancelledError:
            log.info("[Simulator] Stopped loop for %s", patient_id)
            break
        except Exception as e:
            log.error("[Simulator] Error for %s: %s", patient_id, e)

        await asyncio.sleep(TICK_INTERVAL)


async def _patient_discovery_loop():
    """
    Polls the API every PATIENT_POLL_INTERVAL seconds for new patients.
    When a new patient is found, creates a simulator and starts a
    simulation loop for them automatically.
    """
    global _producer
    kafka_cfg = get_kafka_config()

    # Wait for FastAPI to be ready
    await asyncio.sleep(3)
    log.info("[Simulator] Starting patient discovery...")

    # Start shared Kafka producer
    _producer = AIOKafkaProducer(
        **kafka_cfg,
        compression_type="gzip",
        acks="all",
    )

    while True:
        try:
            await _producer.start()
            log.info("[Simulator] Kafka producer connected")
            break
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
                    # Skip if we're already simulating this patient
                    if pid in _sim_tasks and not _sim_tasks[pid].done():
                        continue

                    # New patient! Create simulator and start loop
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
# Vitals Consumer — reads Kafka, POSTs to FastAPI (same as vitals_consumer.py)
# ═══════════════════════════════════════════════════════════════════════════════

async def _vitals_consumer_loop():
    """Consumes vitals.raw, POSTs to /icu/vitals, gets AI risk, publishes to inference.output."""
    kafka_cfg = get_kafka_config()
    await asyncio.sleep(5)  # wait for FastAPI + simulator to be ready
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

                    try:
                        token = await _get_token(client)

                        # 1. POST vitals
                        resp = await client.post(
                            f"{SELF_BASE}/icu/vitals/{patient_id}",
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
                            f"{SELF_BASE}/icu/ai/risk/{patient_id}",
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
                        log.error("API error for %s: %s %s", patient_id, e.response.status_code, e.response.text[:200])
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
# Labs Consumer — reads Kafka, POSTs to FastAPI (same as labs_consumer.py)
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

                    try:
                        token = await _get_token(client)
                        resp = await client.post(
                            f"{SELF_BASE}/icu/labs/{patient_id}",
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
                        log.error("API error for %s: %s %s", patient_id, e.response.status_code, e.response.text[:200])
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
# Public API — call from main.py startup/shutdown
# ═══════════════════════════════════════════════════════════════════════════════

async def start_pipeline():
    """
    Start all pipeline tasks (patient discovery + consumers).
    Call from your FastAPI startup event.

    The patient discovery loop polls your API every PATIENT_POLL_INTERVAL
    seconds and auto-starts simulation for any new patients it finds.
    """
    log.info("═" * 50)
    log.info("Starting embedded Kafka pipeline...")
    log.info("  Kafka: %s", get_kafka_config()["bootstrap_servers"])
    log.info("  Self API: %s", SELF_BASE)
    log.info("  Tick interval: %ds", TICK_INTERVAL)
    log.info("  Patient poll interval: %ds", PATIENT_POLL_INTERVAL)
    log.info("═" * 50)

    _tasks.append(asyncio.create_task(_patient_discovery_loop()))
    _tasks.append(asyncio.create_task(_vitals_consumer_loop()))
    _tasks.append(asyncio.create_task(_labs_consumer_loop()))


async def stop_pipeline():
    """
    Stop all pipeline tasks + per-patient simulation tasks.
    Call from your FastAPI shutdown event.
    """
    global _producer
    log.info("Stopping pipeline tasks...")

    # Cancel per-patient simulation tasks
    for pid, task in _sim_tasks.items():
        task.cancel()
    if _sim_tasks:
        await asyncio.gather(*_sim_tasks.values(), return_exceptions=True)
    _sim_tasks.clear()
    _simulators.clear()

    # Cancel main tasks (discovery + consumers)
    for task in _tasks:
        task.cancel()
    await asyncio.gather(*_tasks, return_exceptions=True)
    _tasks.clear()

    # Stop shared Kafka producer
    if _producer:
        try:
            await _producer.stop()
        except Exception:
            pass
        _producer = None

    log.info("Pipeline stopped.")
