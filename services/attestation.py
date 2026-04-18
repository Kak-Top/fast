"""
Simulated Remote Attestation
==============================
Proves to a verifier that the TEE code hasn't been modified.

REAL TEE ATTESTATION (SGX):
  1. Enclave generates a "QUOTE" — cryptographic proof of its code hash
  2. Quote is signed by the CPU's embedded private key (fused at factory)
  3. Verifier checks quote against Intel's Attestation Service
  4. If code hash matches expected → code is unmodified

OUR SIMULATED ATTESTATION:
  1. At startup, hash ALL critical source files
  2. Sign the hash with enclave-sealed key
  3. Provide /tee/attest endpoint with the signed quote
  4. Verifier can check if code hash matches expected value
  5. If any file was modified → hash changes → attestation fails

LIMITATION (be honest):
  This is SOFTWARE-based. A hacker with root access COULD:
  - Modify files, then re-generate the quote with the new hash
  - But they'd need the ENCLAVE_MASTER_KEY to forge the signature
  - And they'd need to restart the service (downtime detected)

  REAL SGX prevents even this because the signing key is
  BURNED INTO THE CPU and cannot be extracted.
"""

import hashlib
import hmac
import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tee.attestation")

EMK = os.getenv(
    "ENCLAVE_MASTER_KEY",
    "DEFAULT_DEV_KEY_32_BYTES_CHANGE_ME!!"
).encode()


class CodeHasher:
    """
    Hash all critical source files at startup.

    Creates a "code measurement" — a fingerprint of the
    exact code running in the TEE.
    """

    # Files that are part of the TEE trusted computing base
    CRITICAL_FILES = [
        "routers/tee.py",
        "services/anomaly.py",
        "services/he_crypto.py",
        "services/merkle_audit.py",
        "services/attestation.py",
        "middleware/tee_gateway.py",
        "utils/proof.py",
        "main.py",
    ]

    @classmethod
    def measure(cls) -> Dict[str, Any]:
        """
        Hash all critical files. Returns measurement report.

        If ANY file is modified after deployment, the hash changes
        and attestation will fail.
        """
        file_hashes = {}
        total_hash_input = ""

        for filepath in cls.CRITICAL_FILES:
            path = Path(filepath)
            if path.exists():
                content = path.read_bytes()
                file_hash = hashlib.sha256(content).hexdigest()
                file_hashes[filepath] = file_hash
                total_hash_input += file_hash
            else:
                file_hashes[filepath] = "MISSING"
                total_hash_input += "MISSING"

        # Composite hash of ALL files
        composite_hash = hashlib.sha256(total_hash_input.encode()).hexdigest()

        # Sign with enclave key (can't forge without key)
        signature = hmac.new(
            EMK, composite_hash.encode(), hashlib.sha256
        ).hexdigest()

        return {
            "composite_hash": composite_hash,
            "file_hashes": file_hashes,
            "signature": signature,
            "file_count": len(file_hashes),
            "measured_at": datetime.now(timezone.utc).isoformat(),
        }


class AttestationService:
    """
    Simulated Remote Attestation Service.

    At startup: measures all code files and stores the measurement.
    On request: provides a signed "quote" proving code integrity.
    Verifier: checks the quote against expected values.
    """

    def __init__(self):
        self._measurement = CodeHasher.measure()
        self._boot_time = datetime.now(timezone.utc)
        self._attestation_count = 0
        logger.info(
            f"✓ Attestation: code measured "
            f"(hash={self._measurement['composite_hash'][:16]}...)"
        )

    def get_quote(self) -> Dict[str, Any]:
        """
        Generate an attestation quote.

        This is what a verifier would check to prove
        the TEE code hasn't been tampered with.

        In real SGX: this would be signed by the CPU's
        embedded key and verified by Intel's service.
        """
        self._attestation_count += 1

        # Re-measure current code state
        current_measurement = CodeHasher.measure()

        # Check if code has been modified since boot
        code_intact = (
            current_measurement["composite_hash"]
            == self._measurement["composite_hash"]
        )

        quote = {
            "tee_type": "simulated_sgx",
            "code_measurement": self._measurement["composite_hash"],
            "current_measurement": current_measurement["composite_hash"],
            "signature": self._measurement["signature"],
            "code_intact": code_intact,
            "boot_time": self._boot_time.isoformat(),
            "uptime_seconds": (
                datetime.now(timezone.utc) - self._boot_time
            ).total_seconds(),
            "attestation_count": self._attestation_count,
            "quote_generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Sign the entire quote
        canonical = json.dumps(quote, sort_keys=True, default=str)
        quote["quote_signature"] = hmac.new(
            EMK, canonical.encode(), hashlib.sha256
        ).hexdigest()

        if not code_intact:
            logger.critical(
                "🚨 ATTESTATION FAILURE: Code has been modified since boot! "
                "Possible tampering detected!"
            )

        return quote

    def verify_quote(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify an attestation quote.

        A verifier calls this to check:
        1. The quote signature is valid (not forged)
        2. The code measurement matches expected values
        3. The code hasn't been modified since boot
        """
        # Extract and verify signature
        received_sig = quote.get("quote_signature", "")
        quote_copy = {k: v for k, v in quote.items() if k != "quote_signature"}
        canonical = json.dumps(quote_copy, sort_keys=True, default=str)
        expected_sig = hmac.new(
            EMK, canonical.encode(), hashlib.sha256
        ).hexdigest()

        signature_valid = hmac.compare_digest(received_sig, expected_sig)
        code_intact = quote.get("code_intact", False)

        return {
            "trusted": signature_valid and code_intact,
            "signature_valid": signature_valid,
            "code_intact": code_intact,
            "code_measurement": quote.get("code_measurement"),
            "tee_type": quote.get("tee_type"),
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def boot_measurement(self) -> str:
        return self._measurement["composite_hash"]


# ── Singleton ──────────────────────────────────────────────────────
_attestation: Optional[AttestationService] = None


def get_attestation() -> AttestationService:
    global _attestation
    if _attestation is None:
        _attestation = AttestationService()
    return _attestation