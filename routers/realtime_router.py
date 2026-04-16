"""
NEW ENDPOINTS — Add these to your FastAPI application.

HOW TO ADD:
1. Copy this file into your project as `routers/realtime_router.py`
2. In your main.py, add:
       from routers.realtime_router import router as realtime_router, _kafka_inference_listener
       app.include_router(realtime_router)
3. Install deps: pip install aiokafka websockets

This file adds:
  POST /icu/labs/{patient_id}         — new labs ingestion endpoint
  GET  /icu/labs/{patient_id}/latest  — get latest labs for a patient
  WS   /ws/stream/{patient_id}        — real-time push stream to UI
  GET  /ws/status                     — how many clients are connected
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from kafka_config import get_kafka_config

log = logging.getLogger(__name__)

router = APIRouter()

# ─── Re-use your existing auth dependency ─────────────────────────────────────
# Replace this with however your app validates JWT tokens.
# If you have a get_current_user dependency, import and use that instead.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ─── In-memory storage for labs ──────────────────────────────────────────────
# Replace with your database calls if you have a DB layer.
# Structure: { patient_id: { labs_dict, timestamp, flags } }
_latest_labs: dict = {}

# Normal lab ranges for flagging
LAB_RANGES = {
    "glucose": (70, 180, "mg/dL"),
    "creatinine": (0.6, 1.2, "mg/dL"),
    "wbc": (4.5, 11.0, "10^9/L"),
    "lactate": (0.5, 2.0, "mmol/L"),
}


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class LabsReading(BaseModel):
    glucose: float = Field(..., description="Blood glucose in mg/dL")
    creatinine: float = Field(..., description="Creatinine in mg/dL")
    wbc: float = Field(..., description="White blood cell count in 10^9/L")
    lactate: float = Field(..., description="Lactate in mmol/L")


# ─── WebSocket Connection Manager ────────────────────────────────────────────
# (Same manager as ws_broadcaster.py — import that instead if in same project)

from collections import defaultdict
from typing import Dict, Set


class _ConnectionManager:
    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = defaultdict(set)

    async def connect(self, patient_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[patient_id].add(websocket)
        log.info("WS connected: patient=%s total=%d", patient_id, len(self._connections[patient_id]))

    def disconnect(self, patient_id: str, websocket: WebSocket):
        self._connections[patient_id].discard(websocket)

    async def broadcast(self, patient_id: str, payload: dict):
        if patient_id not in self._connections:
            return
        dead = set()
        text = json.dumps(payload)
        for ws in self._connections[patient_id]:
            try:
                await ws.send_text(text)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self._connections[patient_id].discard(ws)

    def total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())

    def status(self) -> dict:
        return {pid: len(ws) for pid, ws in self._connections.items() if ws}


manager = _ConnectionManager()


# ─── Kafka background task ────────────────────────────────────────────────────

# ─── Kafka background task ────────────────────────────────────────────────────

_kafka_task: Optional[asyncio.Task] = None


async def _kafka_inference_listener():
    """
    Background coroutine that reads inference.output from Kafka
    and broadcasts each message to subscribed WebSocket clients.
    
     ENHANCED: Now includes TurboQuant compression metadata for UI badges
    """
    try:
        from aiokafka import AIOKafkaConsumer
    except ImportError:
        log.error("aiokafka not installed. Run: pip install aiokafka")
        return

    kafka_cfg = get_kafka_config()
    consumer = AIOKafkaConsumer(
        "inference.output",
        **kafka_cfg,
        group_id="websocket-broadcaster",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode()),
    )

    while True:
        try:
            await consumer.start()
            log.info("Kafka inference listener started — broadcasting to WebSocket clients")
            async for msg in consumer:
                data = msg.value
                patient_id = data.get("patient_id")
                if not patient_id:
                    continue

                #  NEW: Try to get TurboQuant metadata from cache (if available)
                turboquant_meta = {}
                try:
                    from engine.cache import EncryptedKVCache
                    cache = EncryptedKVCache()
                    encoded = cache.get(patient_id)
                    if encoded:
                        turboquant_meta = {
                            "turboquant_enabled": True,
                            "compression_ratio": encoded["stats"]["compression_ratio"],
                            "vram_saved_percent": encoded["stats"]["vram_saved_percent"],
                            "encoding_latency_ms": encoded["metadata"]["encoding_latency_ms"],
                            "badge_text": f" {encoded['stats']['compression_ratio']} compression",
                            "secure_computation": True
                        }
                except Exception as tq_error:
                    # Safe fallback - don't break broadcast if TurboQuant fails
                    log.debug(f"TurboQuant metadata not available: {tq_error}")
                    turboquant_meta = {"turboquant_enabled": False}

                # Shape the payload for the UI
                ui_payload = {
                    "type": "update",
                    "patient_id": patient_id,
                    "vitals": data.get("vitals", {}),
                    "risk": data.get("risk", {}),
                    "flags": data.get("vitals_response", {}).get("abnormal_flags", []),
                    "is_critical": data.get("vitals_response", {}).get("is_critical", False),
                    "simulator_state": data.get("simulator_state"),
                    "timestamp": data.get("timestamp"),
                    #  ADD TurboQuant metadata for "WOW" UI badges
                    "turboquant": turboquant_meta,
                }

                # Also include latest labs if we have them
                if patient_id in _latest_labs:
                    ui_payload["labs"] = _latest_labs[patient_id]["labs"]

                # Broadcast to WebSocket clients
                await manager.broadcast(patient_id, ui_payload)
                
                # Log for monitoring
                if turboquant_meta.get("turboquant_enabled"):
                    log.debug(
                        " Broadcast %s: risk=%.2f | %s | %s",
                        patient_id,
                        data.get("risk", {}).get("score", 0),
                        turboquant_meta["compression_ratio"],
                        turboquant_meta["badge_text"]
                    )

        except asyncio.CancelledError:
            log.info("Kafka inference listener shutting down")
            break
        except Exception as e:
            log.error("Kafka inference listener error: %s — retrying in 5s", e)
            await asyncio.sleep(5)
        finally:
            try:
                await consumer.stop()
            except Exception:
                pass


# ─── Endpoint 1: POST /icu/labs/{patient_id} ─────────────────────────────────

@router.post(
    "/icu/labs/{patient_id}",
    tags=["Labs"],
    summary="Push a new labs reading",
    description="""
Ingest a new clinical labs reading for a patient.
Called automatically by the Kafka labs consumer — no manual entry needed.
Stores the latest reading and flags out-of-range values.

**Sample Request:**
```json
{
  "glucose": 132,
  "creatinine": 1.2,
  "wbc": 11.2,
  "lactate": 2.2
}
```
""",
)
async def push_labs(
    patient_id: str,
    labs: LabsReading,
    token: str = Depends(oauth2_scheme),
):
    # Flag abnormal values
    abnormal_flags = []
    labs_dict = labs.model_dump()

    for field_name, (low, high, unit) in LAB_RANGES.items():
        value = labs_dict.get(field_name)
        if value is None:
            continue
        if value < low:
            severity = "CRITICAL" if value < low * 0.8 else "WARNING"
            abnormal_flags.append({
                "parameter": field_name,
                "value": value,
                "normal_range": f"{low}–{high} {unit}",
                "direction": "LOW",
                "severity": severity,
            })
        elif value > high:
            severity = "CRITICAL" if value > high * 1.5 else "WARNING"
            abnormal_flags.append({
                "parameter": field_name,
                "value": value,
                "normal_range": f"{low}–{high} {unit}",
                "direction": "HIGH",
                "severity": severity,
            })

    is_critical = any(f["severity"] == "CRITICAL" for f in abnormal_flags)

    # Store latest
    timestamp = datetime.utcnow().isoformat()
    _latest_labs[patient_id] = {
        "labs": labs_dict,
        "abnormal_flags": abnormal_flags,
        "is_critical": is_critical,
        "timestamp": timestamp,
    }

    # Broadcast labs update to any connected WebSocket clients
    await manager.broadcast(patient_id, {
        "type": "labs_update",
        "patient_id": patient_id,
        "labs": labs_dict,
        "abnormal_flags": abnormal_flags,
        "is_critical": is_critical,
        "timestamp": timestamp,
    })

    return {
        "patient_id": patient_id,
        "timestamp": timestamp,
        "labs": labs_dict,
        "is_critical": is_critical,
        "abnormal_flags": abnormal_flags,
    }


# ─── Endpoint 2: GET /icu/labs/{patient_id}/latest ───────────────────────────

@router.get(
    "/icu/labs/{patient_id}/latest",
    tags=["Labs"],
    summary="Get latest labs for a patient",
)
async def get_latest_labs(
    patient_id: str,
    token: str = Depends(oauth2_scheme),
):
    if patient_id not in _latest_labs:
        raise HTTPException(status_code=404, detail=f"No lab results found for patient {patient_id}")
    return {
        "patient_id": patient_id,
        **_latest_labs[patient_id],
    }


# ─── Endpoint 3: WS /ws/stream/{patient_id} ──────────────────────────────────

@router.websocket("/ws/stream/{patient_id}")
async def websocket_patient_stream(
    patient_id: str,
    websocket: WebSocket,
):
    """
    WebSocket endpoint. Your UI connects here once and receives live updates.

    Messages pushed to UI:
      { type: "update", vitals: {...}, risk: {...}, flags: [...], labs: {...} }
      { type: "labs_update", labs: {...}, flags: [...] }
      { type: "ping" }

    Connect from JS:
      const ws = new WebSocket("wss://capstone.dpdns.org/ws/stream/P001");
      ws.onmessage = (e) => { const data = JSON.parse(e.data); updateUI(data); }
    """
    await manager.connect(patient_id, websocket)

    # Send current labs immediately on connect so UI doesn't start blank
    if patient_id in _latest_labs:
        await websocket.send_text(json.dumps({
            "type": "labs_update",
            "patient_id": patient_id,
            **_latest_labs[patient_id],
        }))

    try:
        # Keep-alive ping loop — also detects when client disconnects
        while True:
            await asyncio.sleep(30)
            await websocket.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        manager.disconnect(patient_id, websocket)
    except Exception:
        manager.disconnect(patient_id, websocket)


# ─── Endpoint 4: GET /ws/status ──────────────────────────────────────────────

@router.get("/ws/status", tags=["Realtime"], summary="Active WebSocket connections")
async def ws_status(token: str = Depends(oauth2_scheme)):
    return {
        "total_connections": manager.total_connections(),
        "connections_per_patient": manager.status(),
    }
