from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from dependencies import get_current_user, require_role
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from models import Resource, Patient
from sqlalchemy.future import select

router = APIRouter()

class ResourceUpdate(BaseModel):
    status: str           # "available" | "occupied" | "in_use" | "maintenance"
    patient_id: Optional[str] = None


# ── GET /icu/resources ────────────────────────────────────────────────────────
@router.get("/resources", summary="Get ICU resource summary (beds, ventilators, monitors)")
async def get_resources(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Returns a full overview of all ICU resources grouped by type,
    with availability counts.
    """
    query = await db.execute(select(Resource))
    resources = query.scalars().all()

    # Build summary by type
    summary = {}
    resources_list = []
    
    for r in resources:
        t = r.type
        if t not in summary:
            summary[t] = {}
        s = r.status
        summary[t][s] = summary[t].get(s, 0) + 1
        summary[t]["total"] = summary[t].get("total", 0) + 1
        
        resources_list.append({
            "resource_id": r.resource_id,
            "type": r.type,
            "status": r.status,
            "patient_id": r.patient_id
        })

    return {"summary": summary, "resources": resources_list}


# ── PUT /icu/resources/{resource_id} ──────────────────────────────────────────
@router.put("/resources/{resource_id}", summary="Update a resource status")
async def update_resource(
    resource_id: str,
    body: ResourceUpdate,
    current_user=Depends(require_role("clinician", "admin", "manager")),
    db: AsyncSession = Depends(get_db)
):
    """
    Update the status of any ICU resource (bed, ventilator, monitor).
    """
    query = await db.execute(select(Resource).where(Resource.resource_id == resource_id))
    resource = query.scalar_one_or_none()
    
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    valid_statuses = {"available", "occupied", "in_use", "maintenance"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Choose from: {valid_statuses}")

    resource.status = body.status
    resource.patient_id = body.patient_id
    await db.commit()
    await db.refresh(resource)

    return {"message": "Resource updated", "resource": {
        "resource_id": resource.resource_id,
        "type": resource.type,
        "status": resource.status,
        "patient_id": resource.patient_id
    }}


# ── POST /icu/simulation/whatif ────────────────────────────────────────────────
@router.post("/simulation/whatif", summary="Run a what-if simulation scenario")
async def run_whatif(
    scenario: str,
    extra_beds: int = 0,
    extra_ventilators: int = 0,
    extra_staff: int = 0,
    surge_percent: int = 0,
    current_user=Depends(require_role("admin", "manager", "clinician")),
    db: AsyncSession = Depends(get_db)
):
    """
    Simulates a hypothetical scenario and returns projected capacity impact.
    """
    resources_query = await db.execute(select(Resource))
    resources = resources_query.scalars().all()
    
    beds_available = sum(1 for r in resources if r.type == "bed" and r.status == "available")
    vents_available = sum(1 for r in resources if r.type == "ventilator" and r.status == "available")

    patients_query = await db.execute(select(Patient))
    patients = patients_query.scalars().all()
    patient_count = len(patients)

    projected_beds = beds_available + extra_beds
    projected_vents = vents_available + extra_ventilators
    expected_surge = int(patient_count * (1 + surge_percent / 100))
    shortage = expected_surge - (projected_beds + patient_count - beds_available)

    risk_level = "LOW"
    if shortage > 3:
        risk_level = "CRITICAL"
    elif shortage > 0:
        risk_level = "HIGH"
    elif projected_vents < 2:
        risk_level = "MEDIUM"

    return {
        "scenario": scenario,
        "inputs": {
            "surge_percent": surge_percent,
            "extra_beds": extra_beds,
            "extra_ventilators": extra_ventilators,
            "extra_staff": extra_staff,
        },
        "current_capacity": {
            "beds_available": beds_available,
            "ventilators_available": vents_available,
        },
        "projected_capacity": {
            "beds_available": projected_beds,
            "ventilators_available": projected_vents,
        },
        "expected_patient_surge": expected_surge,
        "projected_shortage": max(shortage, 0),
        "recommendation": (
            f"With a {surge_percent}% surge and {extra_beds} extra beds, "
            f"projected shortage is {max(shortage,0)} beds. "
            + ("Ventilator supply is tight." if projected_vents < 2 else "Ventilators are sufficient.")
        ),
        "risk_level": risk_level,
    }
