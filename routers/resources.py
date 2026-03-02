from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from dependencies import fake_resources_db, get_current_user, require_role

router = APIRouter()

class ResourceUpdate(BaseModel):
    status: str           # "available" | "occupied" | "in_use" | "maintenance"
    patient_id: Optional[str] = None


# ── GET /icu/resources ────────────────────────────────────────────────────────
@router.get("/resources", summary="Get ICU resource summary (beds, ventilators, monitors)")
def get_resources(current_user=Depends(get_current_user)):
    """
    Returns a full overview of all ICU resources grouped by type,
    with availability counts.

    **Sample Response:**
    ```json
    {
      "summary": {
        "beds":        { "total": 3, "available": 1, "occupied": 2 },
        "ventilators": { "total": 2, "available": 1, "in_use": 1 },
        "monitors":    { "total": 2, "available": 0, "in_use": 2 }
      },
      "resources": [
        { "resource_id": "ICU-01",  "type": "bed",        "status": "occupied",  "patient_id": "P001" },
        { "resource_id": "ICU-02",  "type": "bed",        "status": "occupied",  "patient_id": "P002" },
        { "resource_id": "ICU-03",  "type": "bed",        "status": "available", "patient_id": null   },
        { "resource_id": "VENT-01", "type": "ventilator", "status": "in_use",    "patient_id": "P001" },
        { "resource_id": "VENT-02", "type": "ventilator", "status": "available", "patient_id": null   }
      ]
    }
    ```
    """
    resources = list(fake_resources_db.values())

    # Build summary by type
    summary = {}
    for r in resources:
        t = r["type"]
        if t not in summary:
            summary[t] = {}
        s = r["status"]
        summary[t][s] = summary[t].get(s, 0) + 1
        summary[t]["total"] = summary[t].get("total", 0) + 1

    return {"summary": summary, "resources": resources}


# ── PUT /icu/resources/{resource_id} ──────────────────────────────────────────
@router.put("/resources/{resource_id}", summary="Update a resource status")
def update_resource(
    resource_id: str,
    body: ResourceUpdate,
    current_user=Depends(require_role("clinician", "admin", "manager"))
):
    """
    Update the status of any ICU resource (bed, ventilator, monitor).

    **Sample Request Body:**
    ```json
    {
      "status": "occupied",
      "patient_id": "P003"
    }
    ```

    **Sample Response:**
    ```json
    {
      "message": "Resource updated",
      "resource": {
        "resource_id": "ICU-03",
        "type": "bed",
        "status": "occupied",
        "patient_id": "P003"
      }
    }
    ```
    """
    resource = fake_resources_db.get(resource_id)
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    valid_statuses = {"available", "occupied", "in_use", "maintenance"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Choose from: {valid_statuses}")

    resource["status"] = body.status
    resource["patient_id"] = body.patient_id

    return {"message": "Resource updated", "resource": resource}


# ── POST /icu/simulation/whatif ────────────────────────────────────────────────
@router.post("/simulation/whatif", summary="Run a what-if simulation scenario")
def run_whatif(
    scenario: str,
    extra_beds: int = 0,
    extra_ventilators: int = 0,
    extra_staff: int = 0,
    surge_percent: int = 0,
    current_user=Depends(require_role("admin", "manager", "clinician"))
):
    """
    Simulates a hypothetical scenario and returns projected capacity impact.

    **Example:** `?scenario=pandemic_surge&surge_percent=30&extra_beds=5`

    **Sample Response:**
    ```json
    {
      "scenario": "pandemic_surge",
      "inputs": {
        "surge_percent": 30,
        "extra_beds": 5,
        "extra_ventilators": 0,
        "extra_staff": 0
      },
      "current_capacity": {
        "beds_available": 1,
        "ventilators_available": 1
      },
      "projected_capacity": {
        "beds_available": 6,
        "ventilators_available": 1
      },
      "recommendation": "With a 30% surge, adding 5 beds helps but ventilator shortage is a risk.",
      "risk_level": "HIGH"
    }
    ```
    """
    beds_available = sum(1 for r in fake_resources_db.values() if r["type"] == "bed" and r["status"] == "available")
    vents_available = sum(1 for r in fake_resources_db.values() if r["type"] == "ventilator" and r["status"] == "available")

    projected_beds = beds_available + extra_beds
    projected_vents = vents_available + extra_ventilators
    patient_count = len(__import__("dependencies").fake_patients_db)
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
