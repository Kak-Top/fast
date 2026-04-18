from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from datetime import datetime
from dependencies import get_current_user, require_role
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from models import Patient, Vital
from sqlalchemy.future import select
import uuid

# ── TEE Imports ──────────────────────────────────────────
from utils.proof import tee_response
from services.merkle_audit import get_merkle_tree

router = APIRouter()

class AdmitPatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    diagnosis: str
    bed_id: str

# ── Helper: Log to Merkle audit trail ────────────────────
def audit_log(event_type: str, actor: str, data: dict):
    """Log an event to the immutable Merkle audit trail."""
    try:
        merkle = get_merkle_tree()
        result = merkle.add_entry(
            event_type=event_type,
            actor=actor,
            data=data,
        )
        return result
    except Exception as e:
        # Audit logging should NEVER break the route
        print(f"⚠ Merkle audit log failed: {e}")
        return None

# ── Helper: Extract username from request ────────────────
def get_actor(request: Request) -> str:
    """Get the current user from request headers or auth state."""
    try:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            import base64, json
            token = auth_header.split(" ")[1]
            parts = token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                padding = 4 - len(payload) % 4
                payload += "=" * padding
                data = json.loads(base64.b64decode(payload))
                return data.get("sub", data.get("username", "authenticated"))
    except Exception:
        pass
    return request.headers.get("X-User", "anonymous")


# ── POST /icu/patients ───────────────────────────────────
@router.post("/patients", summary="Admit a new patient to ICU (TEE-Sealed)")
async def admit_patient(
    request: Request,
    body: AdmitPatientRequest,
    current_user=Depends(require_role("clinician", "admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Admits a new patient into the ICU and assigns them a bed.
    Response is sealed with HMAC proof and logged to Merkle audit trail.
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

    # Build response data
    response_data = {
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

    # ── TEE: Log to Merkle audit trail ──────────────────
    actor = current_user.get("username", get_actor(request))
    audit_log(
        event_type="PATIENT_ADMIT",
        actor=actor,
        data={
            "patient_id": new_id,
            "name": body.name,
            "bed_id": body.bed_id,
            "diagnosis": body.diagnosis,
            "action": "admit",
        }
    )

    # ── TEE: Seal response with HMAC proof ──────────────
    return tee_response(response_data, request)


# ── GET /icu/patients ────────────────────────────────────
@router.get("/patients", summary="List all ICU patients (TEE-Sealed)")
async def list_patients(
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns all currently admitted ICU patients.
    Response is sealed with HMAC proof and logged to Merkle audit trail.
    """
    query = await db.execute(select(Patient))
    patients = query.scalars().all()

    response_data = {
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

    # ── TEE: Log to Merkle audit trail ──────────────────
    actor = current_user.get("username", get_actor(request)) if isinstance(current_user, dict) else get_actor(request)
    audit_log(
        event_type="PATIENT_LIST",
        actor=actor,
        data={
            "action": "list_all",
            "total_returned": len(patients),
        }
    )

    # ── TEE: Seal response with HMAC proof ──────────────
    return tee_response(response_data, request)


# ── GET /icu/patients/{patient_id} ──────────────────────
@router.get("/patients/{patient_id}", summary="Get patient details + latest vitals (TEE-Sealed)")
async def get_patient(
    patient_id: str,
    request: Request,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns full patient profile plus their most recent vitals reading.
    Response is sealed with HMAC proof and logged to Merkle audit trail.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()

    if not patient:
        # ── TEE: Log failed access attempt ──────────────
        audit_log(
            event_type="PATIENT_READ_FAILED",
            actor=get_actor(request),
            data={
                "patient_id": patient_id,
                "action": "read",
                "reason": "not_found",
            }
        )
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

    response_data = {
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

    # ── TEE: Log to Merkle audit trail ──────────────────
    actor = current_user.get("username", get_actor(request)) if isinstance(current_user, dict) else get_actor(request)
    audit_log(
        event_type="PATIENT_READ",
        actor=actor,
        data={
            "patient_id": patient_id,
            "action": "read",
            "has_vitals": latest_vitals_dict is not None,
        }
    )

    # ── TEE: Seal response with HMAC proof ──────────────
    return tee_response(response_data, request)


# ── DELETE /icu/patients/{patient_id} ────────────────────
@router.delete("/patients/{patient_id}", summary="Discharge patient from ICU (TEE-Sealed)")
async def discharge_patient(
    patient_id: str,
    request: Request,
    current_user=Depends(require_role("clinician", "admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Discharges a patient from the ICU and frees their bed.
    Response is sealed with HMAC proof and logged to Merkle audit trail.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()

    if not patient:
        # ── TEE: Log failed discharge attempt ───────────
        audit_log(
            event_type="PATIENT_DISCHARGE_FAILED",
            actor=get_actor(request),
            data={
                "patient_id": patient_id,
                "action": "discharge",
                "reason": "not_found",
            }
        )
        raise HTTPException(status_code=404, detail="Patient not found")

    # Store patient info before deletion for audit
    patient_snapshot = {
        "patient_id": patient.patient_id,
        "name": patient.name,
        "bed_id": patient.bed_id,
        "status": patient.status,
    }

    await db.delete(patient)
    await db.commit()

    response_data = {
        "message": f"Patient {patient_id} discharged successfully",
        "discharged_by": current_user["username"],
        "discharged_at": datetime.utcnow().isoformat(),
    }

    # ── TEE: Log to Merkle audit trail ──────────────────
    actor = current_user.get("username", get_actor(request))
    audit_log(
        event_type="PATIENT_DISCHARGE",
        actor=actor,
        data={
            "patient_id": patient_id,
            "action": "discharge",
            "patient_snapshot": patient_snapshot,
        }
    )

    # ── TEE: Seal response with HMAC proof ──────────────
    return tee_response(response_data, request)
