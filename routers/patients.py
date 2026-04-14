from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from dependencies import get_current_user, require_role
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from models import Patient, Vital
from sqlalchemy.future import select
import uuid

router = APIRouter()

class AdmitPatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    diagnosis: str
    bed_id: str

# ── POST /icu/patients ────────────────────────────────────────────────────────
@router.post("/patients", summary="Admit a new patient to ICU")
async def admit_patient(
    body: AdmitPatientRequest,
    current_user=Depends(require_role("clinician", "admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Admits a new patient into the ICU and assigns them a bed.
    """
    new_id = f"P-{uuid.uuid4().hex[:6].upper()}"
    patient = Patient(
        patient_id=new_id,
        name=body.name,
        age=body.age,
        gender=body.gender,
        diagnosis=body.diagnosis,
        bed_id=body.bed_id,
        status="stable",
    )
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    
    return {
        "message": "Patient admitted successfully",
        "patient": {
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "diagnosis": patient.diagnosis,
            "bed_id": patient.bed_id,
            "admitted_at": patient.admitted_at.isoformat(),
            "status": patient.status
        }
    }


# ── GET /icu/patients ─────────────────────────────────────────────────────────
@router.get("/patients", summary="List all ICU patients")
async def list_patients(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Returns all currently admitted ICU patients.
    """
    query = await db.execute(select(Patient))
    patients = query.scalars().all()
    
    return {
        "total": len(patients),
        "patients": [
            {
                "patient_id": p.patient_id,
                "name": p.name,
                "age": p.age,
                "diagnosis": p.diagnosis,
                "bed_id": p.bed_id,
                "status": p.status
            } for p in patients
        ]
    }


# ── GET /icu/patients/{patient_id} ────────────────────────────────────────────
@router.get("/patients/{patient_id}", summary="Get patient details + latest vitals")
async def get_patient(patient_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Returns full patient profile plus their most recent vitals reading.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    vitals_query = await db.execute(
        select(Vital).where(Vital.patient_id == patient_id).order_by(Vital.timestamp.desc()).limit(1)
    )
    latest_vital = vitals_query.scalar_one_or_none()

    latest_vitals_dict = None
    if latest_vital:
        latest_vitals_dict = {
            "heart_rate": latest_vital.heart_rate,
            "blood_pressure_sys": latest_vital.blood_pressure_sys,
            "blood_pressure_dia": latest_vital.blood_pressure_dia,
            "spo2": latest_vital.spo2,
            "respiratory_rate": latest_vital.respiratory_rate,
            "temperature": latest_vital.temperature,
            "timestamp": latest_vital.timestamp.isoformat()
        }

    return {
        "patient_id": patient.patient_id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "diagnosis": patient.diagnosis,
        "bed_id": patient.bed_id,
        "status": patient.status,
        "admitted_at": patient.admitted_at.isoformat() if patient.admitted_at else None,
        "latest_vitals": latest_vitals_dict
    }


# ── DELETE /icu/patients/{patient_id} ─────────────────────────────────────────
@router.delete("/patients/{patient_id}", summary="Discharge patient from ICU")
async def discharge_patient(
    patient_id: str,
    current_user=Depends(require_role("clinician", "admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Discharges a patient from the ICU and frees their bed.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    await db.delete(patient)
    await db.commit()
    
    return {
        "message": f"Patient {patient_id} discharged successfully",
        "discharged_by": current_user["username"],
        "discharged_at": datetime.utcnow().isoformat(),
    }
