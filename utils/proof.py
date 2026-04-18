"""
HMAC Proof Utility — Cryptographic Sealing for TEE Operations.
Every TEE result is sealed with an unforgeable HMAC-SHA256 proof.
If anyone tampers with the data, the proof breaks → tampering detected.
NO SIEM dependencies.
"""

import hmac
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Enclave Master Key — shared across all TEE services
# In production: set via environment variable, NEVER hardcode
EMK = os.getenv(
    "ENCLAVE_MASTER_KEY",
    "DEFAULT_DEV_KEY_32_BYTES_CHANGE_ME!!"
).encode()

if len(EMK) < 32:
    import logging
    logging.getLogger("tee.proof").warning(
        "⚠ ENCLAVE_MASTER_KEY is shorter than 32 bytes — set it properly!"
    )


def seal_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Seal data with an unforgeable HMAC-SHA256 proof.

    Input:  {"threat_score": 0.23, "user": "dr.ahmad", ...}
    Output: {
        "data": {...},
        "proof": "a1b2c3d4...",     ← can't be forged
        "sealed_at": "2025-04-15T..."
    }
    """
    canonical = json.dumps(data, sort_keys=True, default=str)
    proof = hmac.new(EMK, canonical.encode(), hashlib.sha256).hexdigest()

    return {
        "data": data,
        "proof": proof,
        "sealed_at": datetime.now(timezone.utc).isoformat()
    }


def verify_seal(sealed: Dict[str, Any]) -> bool:
    """
    Verify a sealed result hasn't been tampered with.

    If someone changes ANY field in "data", the proof becomes invalid.
    Uses constant-time comparison to prevent timing attacks.
    """
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
            "⚠ PROOF VERIFICATION FAILED — POSSIBLE TAMPERING DETECTED"
        )

    return is_valid


def generate_nonce() -> str:
    """Generate a random nonce for additional security."""
    import secrets
    return secrets.token_hex(16)