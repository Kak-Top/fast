from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from dependencies import fake_patients_db, fake_vitals_db, get_current_user, require_role

router = APIRouter()

class AdmitPatientRequest(BaseModel):
    name: str
    age: int
    gender: str
    diagnosis: str
    bed_id: str

# ── POST /icu/patients ────────────────────────────────────────────────────────
@router.post("/patients", summary="Admit a new patient to ICU")
def admit_patient(
    body: AdmitPatientRequest,
    current_user=Depends(require_role("clinician", "admin"))
):
    """
    Admits a new patient into the ICU and assigns them a bed.

    **Sample Request Body:**
    ```json
    {
      "name": "Omar Nasser",
      "age": 72,
      "gender": "Male",
      "diagnosis": "Acute MI",
      "bed_id": "ICU-03"
    }
    ```

    **Sample Response:**
    ```json
    {
      "message": "Patient admitted successfully",
      "patient": {
        "patient_id": "P003",
        "name": "Omar Nasser",
        "age": 72,
        "gender": "Male",
        "diagnosis": "Acute MI",
        "bed_id": "ICU-03",
        "admitted_at": "2025-03-02T10:00:00",
        "status": "stable"
      }
    }
    ```
    """
    new_id = f"P{str(len(fake_patients_db) + 1).zfill(3)}"
    patient = {
        "patient_id": new_id,
        "name": body.name,
        "age": body.age,
        "gender": body.gender,
        "diagnosis": body.diagnosis,
        "bed_id": body.bed_id,
        "admitted_at": datetime.utcnow().isoformat(),
        "status": "stable",
    }
    fake_patients_db[new_id] = patient
    fake_vitals_db[new_id] = []
    return {"message": "Patient admitted successfully", "patient": patient}


# ── GET /icu/patients ─────────────────────────────────────────────────────────
@router.get("/patients", summary="List all ICU patients")
def list_patients(current_user=Depends(get_current_user)):
    """
    Returns all currently admitted ICU patients.

    **Sample Response:**
    ```json
    {
      "total": 2,
      "patients": [
        {
          "patient_id": "P001",
          "name": "Khalid Al-Mansouri",
          "age": 67,
          "diagnosis": "Respiratory Failure",
          "bed_id": "ICU-01",
          "status": "critical"
        }
    ```
    """
    patients = list(fake_patients_db.values())
    return {"total": len(patients), "patients": patients}


# ── GET /icu/patients/{patient_id} ────────────────────────────────────────────
@router.get("/patients/{patient_id}", summary="Get patient details + latest vitals")
def get_patient(patient_id: str, current_user=Depends(get_current_user)):
    """
    Returns full patient profile plus their most recent vitals reading.

    **Sample Response:**
    ```json
    {
      "patient_id": "P001",
      "name": "Khalid Al-Mansouri",
      "age": 67,
      "gender": "Male",
      "diagnosis": "Respiratory Failure",
      "bed_id": "ICU-01",
      "status": "critical",
      "latest_vitals": {
        "heart_rate": 118,
        "blood_pressure_sys": 85,
        "blood_pressure_dia": 52,
        "spo2": 89,
        "respiratory_rate": 28,
        "temperature": 39.1,
        "timestamp": "2025-03-02T08:15:00"
      }
    }
    ```
    """
    patient = fake_patients_db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    vitals_history = fake_vitals_db.get(patient_id, [])
    latest_vitals = vitals_history[-1] if vitals_history else None

    return {**patient, "latest_vitals": latest_vitals}


# ── DELETE /icu/patients/{patient_id} ─────────────────────────────────────────
@router.delete("/patients/{patient_id}", summary="Discharge patient from ICU")
def discharge_patient(
    patient_id: str,
    current_user=Depends(require_role("clinician", "admin"))
):
    """
    Discharges a patient from the ICU and frees their bed.

    **Sample Response:**
    ```json
    {
      "message": "Patient P001 discharged successfully",
      "discharged_by": "dr.ahmad",
      "discharged_at": "2025-03-02T12:00:00"
    }
    ```
    """
    if patient_id not in fake_patients_db:
        raise HTTPException(status_code=404, detail="Patient not found")

    del fake_patients_db[patient_id]
    return {
        "message": f"Patient {patient_id} discharged successfully",
        "discharged_by": current_user["username"],
        "discharged_at": datetime.utcnow().isoformat(),
    }
