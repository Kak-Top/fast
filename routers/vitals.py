import asyncio
import random
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from datetime import datetime
from dependencies import fake_patients_db, fake_vitals_db, get_current_user

router = APIRouter()

# Normal ranges for flagging
NORMAL_RANGES = {
    "heart_rate":         (60, 100),
    "blood_pressure_sys": (90, 140),
    "blood_pressure_dia": (60, 90),
    "spo2":               (95, 100),
    "respiratory_rate":   (12, 20),
    "temperature":        (36.0, 37.5),
}

def is_critical(vitals: dict) -> bool:
    for key, (low, high) in NORMAL_RANGES.items():
        if key in vitals and not (low <= vitals[key] <= high):
            return True
    return False

def flag_abnormal_params(vitals: dict) -> list:
    flags = []
    for key, (low, high) in NORMAL_RANGES.items():
        if key in vitals and not (low <= vitals[key] <= high):
            flags.append({
                "parameter": key,
                "value": vitals[key],
                "normal_range": f"{low}–{high}",
                "severity": "CRITICAL" if abs(vitals[key] - (low+high)/2) > (high-low) else "WARNING"
            })
    return flags

class VitalsReading(BaseModel):
    heart_rate: float
    blood_pressure_sys: float
    blood_pressure_dia: float
    spo2: float
    respiratory_rate: float
    temperature: float


# ── POST /icu/vitals/{patient_id} ─────────────────────────────────────────────
@router.post("/vitals/{patient_id}", summary="Push a new vitals reading")
def push_vitals(
    patient_id: str,
    body: VitalsReading,
    current_user=Depends(get_current_user)
):
    """
    Ingest a new vitals reading for a patient (from IoT sensor or simulation).
    Automatically flags any out-of-range parameters.

    **Sample Request Body:**
    ```json
    {
      "heart_rate": 118,
      "blood_pressure_sys": 85,
      "blood_pressure_dia": 52,
      "spo2": 89,
      "respiratory_rate": 28,
      "temperature": 39.1
    }
    ```

    **Sample Response:**
    ```json
    {
      "patient_id": "P001",
      "timestamp": "2025-03-02T09:00:00",
      "vitals": { ... },
      "is_critical": true,
      "abnormal_flags": [
        {
          "parameter": "spo2",
          "value": 89,
          "normal_range": "95–100",
          "severity": "CRITICAL"
        },
        {
          "parameter": "respiratory_rate",
          "value": 28,
          "normal_range": "12–20",
          "severity": "WARNING"
        }
      ]
    }
    ```
    """
    if patient_id not in fake_patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    reading = body.dict()
    reading["timestamp"] = datetime.utcnow().isoformat()

    fake_vitals_db.setdefault(patient_id, []).append(reading)

    # Update patient status
    status = "critical" if is_critical(reading) else "stable"
    fake_patients_db[patient_id]["status"] = status

    return {
        "patient_id": patient_id,
        "timestamp": reading["timestamp"],
        "vitals": body.dict(),
        "is_critical": is_critical(reading),
        "abnormal_flags": flag_abnormal_params(reading),
    }


# ── GET /icu/vitals/{patient_id}/history ──────────────────────────────────────
@router.get("/vitals/{patient_id}/history", summary="Get historical vitals for a patient")
def get_vitals_history(
    patient_id: str,
    limit: int = 10,
    current_user=Depends(get_current_user)
):
    """
    Returns the last N vitals readings for a patient (default 10).

    **Sample Response:**
    ```json
    {
      "patient_id": "P001",
      "total_readings": 2,
      "readings": [
        {
          "timestamp": "2025-03-02T08:00:00",
          "heart_rate": 112,
          "blood_pressure_sys": 88,
          "spo2": 91,
          "respiratory_rate": 26,
          "temperature": 38.9
        }
      ]
    }
    ```
    """
    if patient_id not in fake_patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    history = fake_vitals_db.get(patient_id, [])
    return {
        "patient_id": patient_id,
        "total_readings": len(history),
        "readings": history[-limit:],
    }


# ── GET /icu/vitals/critical ───────────────────────────────────────────────────
@router.get("/vitals/critical", summary="Get all patients with critical vitals")
def get_critical_patients(current_user=Depends(get_current_user)):
    """
    Returns all ICU patients that currently have one or more out-of-range vital signs.

    **Sample Response:**
    ```json
    {
      "critical_count": 1,
      "patients": [
        {
          "patient_id": "P001",
          "name": "Khalid Al-Mansouri",
          "bed_id": "ICU-01",
          "status": "critical",
          "latest_vitals": { ... },
          "abnormal_flags": [
            { "parameter": "spo2", "value": 89, "normal_range": "95–100", "severity": "CRITICAL" }
          ]
        }
      ]
    }
    ```
    """
    critical = []
    for pid, patient in fake_patients_db.items():
        history = fake_vitals_db.get(pid, [])
        if not history:
            continue
        latest = history[-1]
        flags = flag_abnormal_params(latest)
        if flags:
            critical.append({
                **patient,
                "latest_vitals": latest,
                "abnormal_flags": flags,
            })

    return {"critical_count": len(critical), "patients": critical}


# ── WebSocket /ws/icu/vitals/{patient_id} ─────────────────────────────────────
@router.websocket("/vitals/ws/{patient_id}")
async def vitals_stream(websocket: WebSocket, patient_id: str):
    """
    Live vitals stream for a patient. Sends a new simulated reading every 3 seconds.

    Connect with: `ws://localhost:8000/icu/vitals/ws/P001`

    **Sample streamed message:**
    ```json
    {
      "patient_id": "P001",
      "timestamp": "2025-03-02T09:01:00",
      "heart_rate": 115,
      "blood_pressure_sys": 87,
      "blood_pressure_dia": 54,
      "spo2": 90,
      "respiratory_rate": 27,
      "temperature": 39.0,
      "is_critical": true
    }
    ```
    """
    await websocket.accept()
    try:
        while True:
            # Simulate a reading (replace with real sensor pull)
            reading = {
                "patient_id": patient_id,
                "timestamp": datetime.utcnow().isoformat(),
                "heart_rate":         round(random.uniform(60, 130), 1),
                "blood_pressure_sys": round(random.uniform(80, 150), 1),
                "blood_pressure_dia": round(random.uniform(50, 95), 1),
                "spo2":               round(random.uniform(85, 100), 1),
                "respiratory_rate":   round(random.uniform(10, 32), 1),
                "temperature":        round(random.uniform(36.0, 40.0), 1),
            }
            reading["is_critical"] = is_critical(reading)
            await websocket.send_json(reading)
            await asyncio.sleep(3)
    except WebSocketDisconnect:
        pass
