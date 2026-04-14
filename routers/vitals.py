import asyncio
import random
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from datetime import datetime
from dependencies import get_current_user
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from models import Patient, Vital
from sqlalchemy.future import select

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
        if key in vitals and vitals[key] is not None and not (low <= vitals[key] <= high):
            return True
    return False

def flag_abnormal_params(vitals: dict) -> list:
    flags = []
    for key, (low, high) in NORMAL_RANGES.items():
        if key in vitals and vitals[key] is not None and not (low <= vitals[key] <= high):
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
async def push_vitals(
    patient_id: str,
    body: VitalsReading,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Ingest a new vitals reading for a patient.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    reading_dict = body.dict()
    reading_dict["timestamp"] = datetime.utcnow().isoformat()
    
    # Insert vital
    vital = Vital(
        patient_id=patient_id,
        heart_rate=body.heart_rate,
        blood_pressure_sys=body.blood_pressure_sys,
        blood_pressure_dia=body.blood_pressure_dia,
        spo2=body.spo2,
        respiratory_rate=body.respiratory_rate,
        temperature=body.temperature
    )
    db.add(vital)

    # Update patient status
    status = "critical" if is_critical(reading_dict) else "stable"
    patient.status = status
    await db.commit()

    return {
        "patient_id": patient_id,
        "timestamp": reading_dict["timestamp"],
        "vitals": reading_dict,
        "is_critical": is_critical(reading_dict),
        "abnormal_flags": flag_abnormal_params(reading_dict),
    }


# ── GET /icu/vitals/{patient_id}/history ──────────────────────────────────────
@router.get("/vitals/{patient_id}/history", summary="Get historical vitals for a patient")
async def get_vitals_history(
    patient_id: str,
    limit: int = 10,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns the last N vitals readings for a patient (default 10).
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    if not query.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Patient not found")

    vitals_query = await db.execute(
        select(Vital).where(Vital.patient_id == patient_id).order_by(Vital.timestamp.desc()).limit(limit)
    )
    history = vitals_query.scalars().all()
    history = list(reversed(history)) # Return oldest to newest conceptually if desired, or keep desc.

    readings = []
    for v in history:
        readings.append({
            "timestamp": v.timestamp.isoformat(),
            "heart_rate": v.heart_rate,
            "blood_pressure_sys": v.blood_pressure_sys,
            "blood_pressure_dia": v.blood_pressure_dia,
            "spo2": v.spo2,
            "respiratory_rate": v.respiratory_rate,
            "temperature": v.temperature
        })

    return {
        "patient_id": patient_id,
        "total_readings": len(history),
        "readings": readings,
    }


# ── GET /icu/vitals/critical ───────────────────────────────────────────────────
@router.get("/vitals/critical", summary="Get all patients with critical vitals")
async def get_critical_patients(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Returns all ICU patients that currently have a 'critical' status.
    """
    query = await db.execute(select(Patient).where(Patient.status == "critical"))
    critical_patients = query.scalars().all()

    result = []
    for patient in critical_patients:
        # Fetch their latest vitals
        v_query = await db.execute(select(Vital).where(Vital.patient_id == patient.patient_id).order_by(Vital.timestamp.desc()).limit(1))
        latest = v_query.scalar_one_or_none()
        
        latest_dict = None
        flags = []
        if latest:
            latest_dict = {
                "heart_rate": latest.heart_rate,
                "blood_pressure_sys": latest.blood_pressure_sys,
                "blood_pressure_dia": latest.blood_pressure_dia,
                "spo2": latest.spo2,
                "respiratory_rate": latest.respiratory_rate,
                "temperature": latest.temperature,
                "timestamp": latest.timestamp.isoformat() if latest.timestamp else None
            }
            flags = flag_abnormal_params(latest_dict)

        result.append({
            "patient_id": patient.patient_id,
            "name": patient.name,
            "bed_id": patient.bed_id,
            "status": patient.status,
            "latest_vitals": latest_dict,
            "abnormal_flags": flags
        })

    return {"critical_count": len(result), "patients": result}


# ── WebSocket /ws/icu/vitals/{patient_id} ─────────────────────────────────────
@router.websocket("/vitals/ws/{patient_id}")
async def vitals_stream(websocket: WebSocket, patient_id: str):
    """
    Live vitals stream for a patient.
    """
    await websocket.accept()
    try:
        while True:
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
