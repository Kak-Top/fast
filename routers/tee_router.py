"""
routers/tee_router.py — TEE-Protected Simulation Endpoints
===========================================================
Drop this file into your routers/ folder and register it in main.py:

    from routers.tee_router import router as tee_router
    app.include_router(tee_router)

This exposes:
  POST /icu/simulation/whatif    — TEE-sealed what-if simulation (replaces old one)
  POST /icu/tee/mutate           — Generic TEE state mutation (close_bed, open_bed, etc.)
  GET  /icu/tee/status           — Enclave health + proof prefix
  GET  /icu/tee/verify           — Explicit proof verification check

The old /icu/simulation/whatif in resources.py is superseded by this router.
If you have the old endpoint defined there, comment it out or remove it.

SIEM integration
----------------
Every mutation → SIEM event automatically (POST /siem/events internally).
Every tamper detection → CRITICAL SIEM alert.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, Field

from tee_enclave import (
    TEEEnclave,
    TEETamperError,
    TEENotInitialized,
    TEEOperationRejected,
    get_enclave,
)

log = logging.getLogger("tee_router")

router = APIRouter(prefix="/icu", tags=["TEE Simulation"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ──────────────────────────────────────────────────────────────────────────────
# Import your existing auth dependency
# Replace get_current_user with however you decode JWT in your project.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dependencies import get_current_user, fake_patients_db, fake_resources_db
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False
    log.warning("Could not import dependencies.py — using token-only auth for TEE router.")


def _current_user(token: str = Depends(oauth2_scheme)):
    """Thin wrapper — use your real get_current_user if available."""
    if _HAS_DEPS:
        # This will be called with the real FastAPI dependency injection
        pass
    # Return a minimal user object for type-checking in this file
    return token  # will be replaced by DI


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic request models
# ──────────────────────────────────────────────────────────────────────────────

class WhatIfRequest(BaseModel):
    scenario: str = Field(..., examples=["pandemic_surge", "maintenance", "staff_shortage"])
    surge_percent: int = Field(0, ge=0, le=300, description="Percent increase in patient load")
    extra_beds: int = Field(0, ge=0, le=100)
    extra_ventilators: int = Field(0, ge=0, le=100)
    extra_staff: int = Field(0, ge=0, le=500)


class TEEMutationRequest(BaseModel):
    operation: str = Field(
        ...,
        examples=["close_bed", "open_bed", "close_resource", "add_resource"],
    )
    params: Dict[str, Any] = Field(
        ...,
        description="Operation-specific parameters",
        examples=[{"resource_id": "ICU-01", "reason": "maintenance"}],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SIEM helper — fire and forget
# ──────────────────────────────────────────────────────────────────────────────

async def _siem_event(
    request: Request,
    event_type: str,
    description: str,
    severity: str,
    user_id: Optional[str] = None,
    extra: Optional[Dict] = None,
):
    """
    Non-blocking SIEM event push.
    Uses the same /siem/events endpoint your existing SIEM router exposes.
    """
    try:
        base = str(request.base_url).rstrip("/")
        payload = {
            "event_type":  event_type,
            "source_ip":   request.client.host if request.client else "unknown",
            "user_id":     user_id,
            "resource":    str(request.url.path),
            "description": description,
            "severity":    severity,
            **(extra or {}),
        }
        # Fire-and-forget using a short timeout — don't block the response
        async with httpx.AsyncClient(timeout=3.0) as client:
            # We need an admin token — get it from the enclave's internal auth
            # For simplicity we call the internal endpoint directly via localhost
            import os
            port = os.getenv("PORT", "10000")
            internal_base = os.getenv("SELF_BASE", f"http://127.0.0.1:{port}")
            resp = await client.post(
                f"{internal_base}/auth/login",
                data={"username": "admin", "password": os.getenv("API_PASSWORD", "admin123")},
            )
            if resp.status_code == 200:
                tok = resp.json().get("access_token", "")
                await client.post(
                    f"{internal_base}/siem/events",
                    json=payload,
                    headers={"Authorization": f"Bearer {tok}"},
                    timeout=3.0,
                )
    except Exception as e:
        log.warning("SIEM push failed (non-fatal): %s", e)


# ──────────────────────────────────────────────────────────────────────────────
# Validators (run INSIDE the enclave boundary)
# ──────────────────────────────────────────────────────────────────────────────

def _whatif_validator(
    operation: str,
    params: Dict[str, Any],
    state: Dict[str, Any],
) -> Tuple[bool, str]:
    """
    Safety rules checked before any what-if mutation is applied.
    Return (True, "OK") to allow, (False, reason) to reject.
    """
    resources = state.get("resources", {})

    if operation in ("close_bed", "close_resource"):
        rid = params.get("resource_id", "")
        # Rule 1: resource must exist
        if rid not in resources:
            return False, f"Resource '{rid}' does not exist in sealed state."
        # Rule 2: cannot close a bed with a patient in it
        res = resources[rid]
        if res.get("type") == "bed" and res.get("patient_id"):
            return False, f"Cannot close bed '{rid}' — patient {res['patient_id']} is assigned."
        # Rule 3: must keep at least 1 available bed
        available_beds = [
            r for r in resources.values()
            if r.get("type") == "bed" and r.get("status") == "available" and r.get("id") != rid
        ]
        if len(available_beds) < 1 and resources.get(rid, {}).get("type") == "bed":
            return False, "Cannot close last available ICU bed — patient safety constraint."

    if operation == "simulate_whatif":
        surge = params.get("surge_percent", 0)
        if surge > 300:
            return False, "Surge percentage cannot exceed 300% in simulation."

    return True, "OK"


# ──────────────────────────────────────────────────────────────────────────────
# Route: POST /icu/simulation/whatif  (TEE-protected, replaces old one)
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/simulation/whatif",
    summary="Run a TEE-sealed what-if simulation scenario",
    description="""
Simulates a hypothetical ICU scenario inside the Trusted Execution Enclave.

The simulation logic and state are cryptographically sealed with HMAC-SHA256.
Even if FastAPI is compromised, the proof lets the frontend detect forged results.

**Returns a sealed result envelope:**
```json
{
  "success": true,
  "operation": "simulate_whatif",
  "result": { "scenario": "...", "risk_level": "HIGH", ... },
  "proof": "abc123def...",
  "version": 5,
  "sealed_at": "2025-04-13T14:32:15Z"
}
```

The `proof` field is an HMAC-SHA256 signature over the entire state.
If the server is hacked and results are forged, the proof will be invalid.
""",
)
async def whatif_simulation(
    body: WhatIfRequest,
    request: Request,
    current_user=Depends(get_current_user) if _HAS_DEPS else Depends(oauth2_scheme),
):
    enclave: TEEEnclave = get_enclave()

    # Role check
    if _HAS_DEPS:
        user_id = getattr(current_user, "user_id", str(current_user))
        username = getattr(current_user, "username", str(current_user))
        role = getattr(current_user, "role", "")
        if role not in ("manager", "admin"):
            await _siem_event(
                request, "TEE_UNAUTHORIZED", 
                f"User '{username}' (role={role}) attempted what-if simulation",
                "WARNING", user_id,
            )
            raise HTTPException(
                status_code=403,
                detail="Only managers and admins can run what-if simulations.",
            )
    else:
        user_id = "unknown"
        username = "unknown"

    params = {
        **body.model_dump(),
        "user_id": user_id,
    }

    try:
        envelope = enclave.mutate(
            operation="simulate_whatif",
            params=params,
            validator=_whatif_validator,
        )

        await _siem_event(
            request, "TEE_WHATIF_SIMULATION",
            f"What-if simulation '{body.scenario}' by '{username}' "
            f"— risk={envelope['result']['risk_level']} proof={envelope['proof'][:16]}…",
            "INFO", user_id,
            extra={"state_proof": envelope["proof"], "enclave_version": envelope["version"]},
        )

        return envelope

    except TEETamperError as e:
        log.error("🚨 TEE TAMPER DETECTED: %s", e)
        await _siem_event(
            request, "TEE_TAMPER_DETECTED",
            f"State integrity failure during what-if simulation: {e}",
            "CRITICAL", user_id,
        )
        raise HTTPException(status_code=500, detail="State integrity check failed. Security team has been notified.")

    except TEEOperationRejected as e:
        await _siem_event(
            request, "TEE_OPERATION_REJECTED",
            f"What-if scenario rejected: {e}",
            "WARNING", user_id,
        )
        raise HTTPException(status_code=400, detail=str(e))

    except TEENotInitialized:
        raise HTTPException(status_code=503, detail="TEE enclave not yet initialized. Try again in a few seconds.")


# ──────────────────────────────────────────────────────────────────────────────
# Route: POST /icu/tee/mutate  (generic TEE state mutation)
# ──────────────────────────────────────────────────────────────────────────────

@router.post(
    "/tee/mutate",
    summary="Apply a validated TEE state mutation (admin only)",
    description="""
Apply a mutation to the sealed digital twin state.

**Supported operations:**
- `close_bed` — params: `{resource_id, reason}`
- `open_bed` — params: `{resource_id}`
- `close_resource` — params: `{resource_id, resource_type, reason}`
- `add_resource` — params: `{resource: {id, type, status, ...}}`

All mutations are validated and HMAC-sealed before and after.
""",
)
async def tee_mutate(
    body: TEEMutationRequest,
    request: Request,
    current_user=Depends(get_current_user) if _HAS_DEPS else Depends(oauth2_scheme),
):
    enclave: TEEEnclave = get_enclave()

    if _HAS_DEPS:
        user_id  = getattr(current_user, "user_id",  str(current_user))
        username = getattr(current_user, "username", str(current_user))
        role     = getattr(current_user, "role", "")
        if role not in ("admin", "manager"):
            raise HTTPException(status_code=403, detail="Admin or manager role required.")
    else:
        user_id  = "unknown"
        username = "unknown"

    params = {**body.params, "user_id": user_id}

    try:
        envelope = enclave.mutate(
            operation=body.operation,
            params=params,
            validator=_whatif_validator,
        )

        await _siem_event(
            request, "TEE_STATE_MUTATION",
            f"TEE mutation '{body.operation}' by '{username}' — proof={envelope['proof'][:16]}…",
            "INFO", user_id,
            extra={"state_proof": envelope["proof"], "enclave_version": envelope["version"]},
        )

        return envelope

    except TEETamperError as e:
        log.error("🚨 TEE TAMPER DETECTED during mutation: %s", e)
        await _siem_event(
            request, "TEE_TAMPER_DETECTED",
            f"State integrity failure during mutation '{body.operation}': {e}",
            "CRITICAL", user_id,
        )
        raise HTTPException(status_code=500, detail="State integrity failure. Security team notified.")

    except TEEOperationRejected as e:
        raise HTTPException(status_code=400, detail=str(e))

    except TEENotInitialized:
        raise HTTPException(status_code=503, detail="TEE enclave not initialized.")


# ──────────────────────────────────────────────────────────────────────────────
# Route: GET /icu/tee/status
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/tee/status",
    summary="TEE enclave health and current proof",
    description="""
Returns enclave health information without exposing any patient data.

```json
{
  "initialized": true,
  "version": 7,
  "proof_prefix": "a1b2c3d4e5f6g7h8…",
  "sealed_at": "2025-04-13T14:32:15Z",
  "tamper_attempts": 0
}
```
""",
)
async def tee_status(
    current_user=Depends(get_current_user) if _HAS_DEPS else Depends(oauth2_scheme),
):
    enclave: TEEEnclave = get_enclave()
    return enclave.status()


# ──────────────────────────────────────────────────────────────────────────────
# Route: GET /icu/tee/verify
# ──────────────────────────────────────────────────────────────────────────────

@router.get(
    "/tee/verify",
    summary="Explicitly verify sealed state integrity",
    description="""
Runs the HMAC proof verification right now.
Returns `{valid: true}` if state is intact, or triggers a CRITICAL SIEM alert if tampered.
""",
)
async def tee_verify(
    request: Request,
    current_user=Depends(get_current_user) if _HAS_DEPS else Depends(oauth2_scheme),
):
    enclave: TEEEnclave = get_enclave()

    if _HAS_DEPS:
        user_id  = getattr(current_user, "user_id",  str(current_user))
    else:
        user_id = "unknown"

    try:
        enclave.verify_and_read()
        return {
            "valid":      True,
            "version":    enclave.get_version(),
            "proof":      enclave.get_proof(),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
    except TEETamperError as e:
        log.error("🚨 Explicit verify failed: %s", e)
        await _siem_event(
            request, "TEE_TAMPER_DETECTED",
            f"Explicit verify detected tampering: {e}",
            "CRITICAL", user_id,
        )
        raise HTTPException(status_code=500, detail="State integrity check FAILED.")
    except TEENotInitialized:
        raise HTTPException(status_code=503, detail="TEE enclave not initialized.")
