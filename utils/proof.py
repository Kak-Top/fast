"""HMAC Proof Utility - Cryptographic Sealing for TEE Operations."""

import hmac
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from starlette.requests import Request

EMK = os.getenv(
    "ENCLAVE_MASTER_KEY",
    "DEFAULT_DEV_KEY_32_BYTES_CHANGE_ME!!"
).encode()

if len(EMK) < 32:
    import logging
    logging.getLogger("tee.proof").warning(
        "ENCLAVE_MASTER_KEY is shorter than 32 bytes - set it properly!"
    )


def seal_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Seal data with an unforgeable HMAC-SHA256 proof."""
    canonical = json.dumps(data, sort_keys=True, default=str)
    proof = hmac.new(EMK, canonical.encode(), hashlib.sha256).hexdigest()
    return {
        "data": data,
        "proof": proof,
        "sealed_at": datetime.now(timezone.utc).isoformat()
    }


def verify_seal(sealed: Dict[str, Any]) -> bool:
    """Verify a sealed result hasn't been tampered with."""
    data = sealed.get("data", sealed.get("model_output", {}))
    received_proof = sealed.get("proof", "")
    if not data or not received_proof:
        return False
    canonical = json.dumps(data, sort_keys=True, default=str)
    expected_proof = hmac.new(EMK, canonical.encode(), hashlib.sha256).hexdigest()
    is_valid = hmac.compare_digest(received_proof, expected_proof)
    if not is_valid:
        import logging
        logging.getLogger("tee.proof").warning(
            "PROOF VERIFICATION FAILED - POSSIBLE TAMPERING DETECTED"
        )
    return is_valid


def tee_response(data: Any, request: Request = None) -> Dict[str, Any]:
    """
    Wrap ANY response data in a TEE-protected envelope.
    Merges data fields at top level so frontend still works
    (response.patients still accessible), plus adds TEE fields.
    """
    from services.merkle_audit import get_merkle_tree

    # Seal the data with HMAC proof
    sealed = seal_data(data)

    # Build response: merge original data fields at top level
    result = {}
    if isinstance(data, dict):
        result.update(data)

    # Add TEE fields
    result["proof"] = sealed["proof"]
    result["sealed_at"] = sealed["sealed_at"]

    # Get current Merkle root
    try:
        merkle = get_merkle_tree()
        result["merkle_root"] = merkle.root_hash
        result["audit_entry_count"] = merkle.entry_count
    except Exception:
        result["merkle_root"] = "unavailable"
        result["audit_entry_count"] = 0

    # Get threat score from gateway middleware
    if request and hasattr(request.state, "threat_score"):
        result["threat_score"] = request.state.threat_score
        result["threat_type"] = request.state.threat_assessment["model_output"]["threat_type"]
    else:
        result["threat_score"] = None
        result["threat_type"] = None

    return result


def generate_nonce() -> str:
    """Generate a random nonce for additional security."""
    return secrets.token_hex(16)
