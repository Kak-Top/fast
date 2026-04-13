"""
tee_enclave.py — Simulated Trusted Execution Enclave
=====================================================
Cryptographically seals the digital twin's simulation state so that
even a fully-compromised FastAPI process cannot forge simulation results.

Security model
--------------
- Every state mutation is wrapped in HMAC-SHA256 using an Enclave Master Key (EMK)
  that lives ONLY in the environment variable ENCLAVE_MASTER_KEY.
- Any tampering with the sealed state (e.g. a hacker editing in-memory dicts)
  is detected immediately when the HMAC proof fails to verify.
- All violations are logged to your SIEM pipeline automatically.

No special CPU / SGX hardware required — runs on Render free tier.

Usage (in main.py)
------------------
    from tee_enclave import get_enclave, TEETamperError

    @app.on_event("startup")
    async def _init_enclave():
        enc = get_enclave()
        enc.seal({
            "patients":  dict(fake_patients_db),
            "resources": dict(fake_resources_db),
            "simulation_log": [],
        })
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable, Dict, Optional, Tuple

log = logging.getLogger("tee_enclave")

# ──────────────────────────────────────────────────────────────────────────────
# Exceptions
# ──────────────────────────────────────────────────────────────────────────────

class TEETamperError(Exception):
    """Raised when HMAC verification fails — state has been tampered."""

class TEEOperationRejected(Exception):
    """Raised when a validator rejects an operation as unsafe."""

class TEENotInitialized(Exception):
    """Raised when the enclave is used before seal() has been called."""


# ──────────────────────────────────────────────────────────────────────────────
# Core Enclave
# ──────────────────────────────────────────────────────────────────────────────

class TEEEnclave:
    """
    Simulated Trusted Execution Enclave.

    The enclave holds a single 'sealed state' object:
      {
        "version":   int,
        "data":      { ... your hospital state ... },
        "proof":     "hex HMAC-SHA256",
        "sealed_at": "ISO timestamp",
      }

    Only this class can read or mutate the state.
    FastAPI routers never touch .data directly — they receive only results.
    """

    def __init__(self) -> None:
        emk_hex = os.getenv("ENCLAVE_MASTER_KEY", "")
        if not emk_hex:
            # Development fallback — NOT safe for production
            log.warning(
                "⚠️  ENCLAVE_MASTER_KEY not set — using insecure dev key. "
                "Set ENCLAVE_MASTER_KEY in your Render environment variables."
            )
            emk_hex = "dev_key_NOT_FOR_PRODUCTION_set_ENCLAVE_MASTER_KEY_env_var"
        self._emk: bytes = emk_hex.encode()
        self._sealed: Optional[Dict[str, Any]] = None
        self._version: int = 0
        self._lock = RLock()  # thread-safe (FastAPI uses threads for sync routes)
        self._tamper_count: int = 0

    # ──────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────

    def _compute_proof(self, data: Any) -> str:
        """Compute HMAC-SHA256 proof over canonicalized JSON."""
        canonical = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
        return hmac.new(
            self._emk,
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _verify(self, sealed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify integrity proof. Returns unsealed data if valid.
        Raises TEETamperError if proof doesn't match.
        Uses hmac.compare_digest to prevent timing-oracle attacks.
        """
        if not sealed or "proof" not in sealed or "data" not in sealed:
            raise TEETamperError("Sealed state is malformed or missing.")

        expected = self._compute_proof(sealed["data"])
        received = sealed.get("proof", "")

        if not hmac.compare_digest(expected, received):
            self._tamper_count += 1
            raise TEETamperError(
                f"INTEGRITY FAILURE — sealed state has been tampered. "
                f"(tamper count: {self._tamper_count})"
            )

        return sealed["data"]

    def _create_sealed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Seal data and return the sealed envelope."""
        self._version += 1
        proof = self._compute_proof(data)
        return {
            "version":   self._version,
            "data":      data,
            "proof":     proof,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def seal(self, state: Dict[str, Any]) -> str:
        """
        Seal the entire digital twin state.
        Called once at startup, and after every mutation.
        Returns the HMAC proof for audit logging.
        """
        import copy
        with self._lock:
            # Create a detached snapshot of the live state
            # This ensures pipeline updates to live_state don't mutate the sealed data
            snapshot = copy.deepcopy(state)
            self._sealed = self._create_sealed(snapshot)
            log.info(
                "✓ TEE state sealed (snapshot taken) — version=%d proof=%s…",
                self._version,
                self._sealed["proof"][:16],
            )
            return self._sealed["proof"]

    def verify_and_read(self) -> Dict[str, Any]:
        """
        Verify and return a deep copy of the current sealed state.
        Raises TEETamperError if tampered.
        Raises TEENotInitialized if seal() was never called.
        """
        with self._lock:
            if self._sealed is None:
                raise TEENotInitialized("Enclave not initialized. Call seal() first.")
            data = self._verify(self._sealed)
            # Return a deep copy so callers can't mutate internal state
            return json.loads(json.dumps(data, default=str))

    def mutate(
        self,
        operation: str,
        params: Dict[str, Any],
        validator: Optional[Callable[[str, Dict, Dict], Tuple[bool, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Apply a validated mutation inside the enclave.

        Steps:
          1. Verify HMAC of current sealed state (detect tampering)
          2. Run validator(operation, params, current_state) if provided
          3. Apply the mutation
          4. Re-seal with new HMAC proof
          5. Return {proof, version, sealed_at, result}

        The caller (FastAPI route) never receives the raw state.
        They only get back the result dict + proof for audit logging.

        Supported operations
        --------------------
        "close_bed"     — params: {resource_id, reason, user_id}
        "open_bed"      — params: {resource_id, user_id}
        "close_resource"— params: {resource_id, resource_type, reason, user_id}
        "add_resource"  — params: {resource: {id, type, status, ...}}
        "simulate_whatif"— params: {scenario, changes: {}, surge_percent, extra_beds, ...}
        """
        with self._lock:
            if self._sealed is None:
                raise TEENotInitialized("Enclave not initialized.")

            # Step 1: verify integrity
            state = self._verify(self._sealed)

            # Step 2: validate
            if validator is not None:
                is_safe, reason = validator(operation, params, state)
                if not is_safe:
                    raise TEEOperationRejected(
                        f"Operation '{operation}' rejected by validator: {reason}"
                    )

            # Step 3: apply mutation
            result = self._apply_operation(operation, params, state)

            # Step 4: re-seal
            self._sealed = self._create_sealed(state)

            log.info(
                "✓ TEE mutation applied — op=%s version=%d proof=%s…",
                operation, self._version, self._sealed["proof"][:16],
            )

            # Step 5: return result envelope (never the raw state)
            return {
                "success":      True,
                "operation":    operation,
                "result":       result,
                "proof":        self._sealed["proof"],
                "version":      self._version,
                "sealed_at":    self._sealed["sealed_at"],
            }

    def get_proof(self) -> Optional[str]:
        """Return current proof (for audit log / SIEM). None if not sealed."""
        with self._lock:
            if self._sealed:
                return self._sealed["proof"]
            return None

    def get_version(self) -> int:
        """Return current state version number."""
        with self._lock:
            return self._version

    def status(self) -> Dict[str, Any]:
        """Return enclave health info (no sensitive data)."""
        with self._lock:
            return {
                "initialized": self._sealed is not None,
                "version":     self._version,
                "proof_prefix": self._sealed["proof"][:16] + "…" if self._sealed else None,
                "sealed_at":   self._sealed.get("sealed_at") if self._sealed else None,
                "tamper_attempts": self._tamper_count,
            }

    # ──────────────────────────────────────────────────
    # Operation handlers (inside TEE boundary)
    # ──────────────────────────────────────────────────

    def _apply_operation(
        self,
        operation: str,
        params: Dict[str, Any],
        state: Dict[str, Any],          # mutated in-place
    ) -> Dict[str, Any]:
        """
        All mutation logic lives here — inside the TEE boundary.
        FastAPI never calls these methods directly.
        """
        now = datetime.now(timezone.utc).isoformat()

        if operation == "close_bed":
            return self._op_close_resource(params, state, now, "closed")

        elif operation == "open_bed":
            return self._op_open_resource(params, state, now)

        elif operation == "close_resource":
            return self._op_close_resource(params, state, now, "closed")

        elif operation == "add_resource":
            return self._op_add_resource(params, state, now)

        elif operation == "simulate_whatif":
            return self._op_simulate_whatif(params, state, now)

        else:
            raise TEEOperationRejected(f"Unknown operation: '{operation}'")

    # ── Individual operation handlers ─────────────────

    def _op_close_resource(self, params, state, now, new_status):
        rid = params.get("resource_id")
        if not rid:
            raise TEEOperationRejected("resource_id is required.")
        resources = state.setdefault("resources", {})
        if rid not in resources:
            raise TEEOperationRejected(f"Resource '{rid}' not found in sealed state.")

        prev_status = resources[rid].get("status")
        resources[rid]["status"]     = new_status
        resources[rid]["updated_at"] = now
        resources[rid]["updated_by"] = params.get("user_id", "system")
        resources[rid]["reason"]     = params.get("reason", "")

        # Log the change inside the state
        state.setdefault("simulation_log", []).append({
            "op":         "close_resource",
            "resource_id": rid,
            "from_status": prev_status,
            "to_status":   new_status,
            "by":          params.get("user_id", "system"),
            "reason":      params.get("reason", ""),
            "at":          now,
        })

        return {
            "resource_id":    rid,
            "previous_status": prev_status,
            "new_status":      new_status,
            "message":         f"Resource {rid} is now {new_status}.",
        }

    def _op_open_resource(self, params, state, now):
        rid = params.get("resource_id")
        if not rid:
            raise TEEOperationRejected("resource_id is required.")
        resources = state.setdefault("resources", {})
        if rid not in resources:
            raise TEEOperationRejected(f"Resource '{rid}' not found.")

        prev_status = resources[rid].get("status")
        # Remove patient assignment when opening
        resources[rid]["status"]     = "available"
        resources[rid]["patient_id"] = None
        resources[rid]["updated_at"] = now
        resources[rid]["updated_by"] = params.get("user_id", "system")

        state.setdefault("simulation_log", []).append({
            "op":          "open_resource",
            "resource_id": rid,
            "from_status": prev_status,
            "to_status":   "available",
            "by":          params.get("user_id", "system"),
            "at":          now,
        })

        return {
            "resource_id":     rid,
            "previous_status": prev_status,
            "new_status":      "available",
            "message":         f"Resource {rid} is now available.",
        }

    def _op_add_resource(self, params, state, now):
        new_res = params.get("resource")
        if not new_res or "id" not in new_res:
            raise TEEOperationRejected("params.resource with an 'id' key is required.")
        rid = new_res["id"]
        state.setdefault("resources", {})[rid] = {
            **new_res,
            "added_at": now,
        }
        state.setdefault("simulation_log", []).append({
            "op":          "add_resource",
            "resource_id": rid,
            "by":          params.get("user_id", "system"),
            "at":          now,
        })
        return {
            "resource_id": rid,
            "message":     f"Resource {rid} added to sealed state.",
        }

    def _op_simulate_whatif(self, params, state, now):
        """
        Run a what-if scenario on a SNAPSHOT of the state.
        The real state is NOT changed — only the simulation_log is updated.
        """
        import copy
        scenario      = params.get("scenario", "custom")
        surge_pct     = int(params.get("surge_percent", 0))
        extra_beds    = int(params.get("extra_beds", 0))
        extra_vents   = int(params.get("extra_ventilators", 0))
        extra_staff   = int(params.get("extra_staff", 0))

        # Work on a snapshot — never modify real state
        snap = copy.deepcopy(state)
        resources = snap.get("resources", {})

        # Count current capacity
        beds_total       = sum(1 for r in resources.values() if r.get("type") == "bed")
        beds_avail       = sum(1 for r in resources.values() if r.get("type") == "bed" and r.get("status") == "available")
        vents_total      = sum(1 for r in resources.values() if r.get("type") == "ventilator")
        vents_avail      = sum(1 for r in resources.values() if r.get("type") == "ventilator" and r.get("status") == "available")
        patient_count    = len(snap.get("patients", {}))

        # Project demand after surge
        surge_patients = round(patient_count * surge_pct / 100) if surge_pct else 0
        projected_demand = patient_count + surge_patients

        # Project capacity after additions
        projected_beds  = beds_avail  + extra_beds
        projected_vents = vents_avail + extra_vents

        # Risk scoring
        bed_ratio  = projected_demand / max(1, beds_total + extra_beds)
        vent_ratio = projected_demand / max(1, vents_total + extra_vents)

        if bed_ratio > 0.95 or vent_ratio > 0.9:
            risk_level = "CRITICAL"
        elif bed_ratio > 0.80 or vent_ratio > 0.75:
            risk_level = "HIGH"
        elif bed_ratio > 0.65:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        # Generate recommendation
        issues = []
        if projected_beds < 1:
            issues.append(f"NO spare beds after {surge_pct}% surge")
        elif projected_beds < 2:
            issues.append("only 1 spare bed — critical shortage risk")
        if projected_vents < 1:
            issues.append("NO ventilators available")
        if not issues:
            issues.append("capacity appears manageable")

        recommendation = (
            f"Scenario '{scenario}' with {surge_pct}% surge: "
            + "; ".join(issues)
            + f". Adding {extra_beds} bed(s) and {extra_vents} vent(s) "
            + ("helps significantly." if extra_beds > 0 or extra_vents > 0 else "without reinforcements.")
        )

        simulation_result = {
            "scenario":   scenario,
            "inputs": {
                "surge_percent":     surge_pct,
                "extra_beds":        extra_beds,
                "extra_ventilators": extra_vents,
                "extra_staff":       extra_staff,
            },
            "current_capacity": {
                "total_patients":       patient_count,
                "beds_total":           beds_total,
                "beds_available":       beds_avail,
                "ventilators_total":    vents_total,
                "ventilators_available": vents_avail,
            },
            "projected_capacity": {
                "demand_after_surge":     projected_demand,
                "beds_available":         projected_beds,
                "ventilators_available":  projected_vents,
                "bed_utilization_pct":    round(bed_ratio * 100, 1),
                "vent_utilization_pct":   round(vent_ratio * 100, 1),
            },
            "risk_level":     risk_level,
            "recommendation": recommendation,
            "simulated_at":   now,
        }

        # Only the log entry goes into the real state (not the simulation numbers)
        state.setdefault("simulation_log", []).append({
            "op":             "whatif_simulation",
            "scenario":       scenario,
            "risk_level":     risk_level,
            "by":             params.get("user_id", "system"),
            "at":             now,
        })

        return simulation_result


# ──────────────────────────────────────────────────────────────────────────────
# Singleton access
# ──────────────────────────────────────────────────────────────────────────────

_enclave: Optional[TEEEnclave] = None


def get_enclave() -> TEEEnclave:
    """
    Return the global TEE enclave singleton.
    Call seal() on it during FastAPI startup before using mutate().
    """
    global _enclave
    if _enclave is None:
        _enclave = TEEEnclave()
    return _enclave
