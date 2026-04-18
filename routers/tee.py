"""
TEE API Router
===============
All TEE endpoints for your Hospital Digital Twin.
Frontend developer: use https://capstone.dpdns.org/tee/...
NO SIEM dependencies.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("tee.router")

router = APIRouter(prefix="/tee", tags=["TEE — Trusted Execution Environment"])


# ═══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════

class AnomalyRequest(BaseModel):
    endpoint: str = Field(..., description="API endpoint path")
    method: str = Field(default="GET", description="HTTP method")
    user: str = Field(default="unknown", description="Username")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp",
    )
    source_ip: Optional[str] = Field(default=None, description="Client IP")
    request_data: Dict[str, Any] = Field(default_factory=dict)


class VitalsRequest(BaseModel):
    heart_rate: float = Field(..., description="Heart rate (bpm)")
    blood_pressure_systolic: float = Field(..., description="Systolic BP (mmHg)")
    blood_pressure_diastolic: float = Field(..., description="Diastolic BP (mmHg)")
    spo2: float = Field(..., description="Oxygen saturation (%)")
    temperature: float = Field(default=36.6, description="Body temp (°C)")
    respiratory_rate: float = Field(default=16, description="Breaths per min")


class EncryptedDataRequest(BaseModel):
    encrypted: str = Field(..., description="Base64-encoded encrypted data")
    mode: str = Field(default="aes_256", description="Encryption mode used")
    feature_names: List[str] = Field(default_factory=list)
    feature_count: int = Field(default=0)


class VerifyProofRequest(BaseModel):
    model_output: Dict[str, Any]
    proof: str
    sealed_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.post(
    "/detect",
    summary="Run Anomaly Detection",
    description="Analyze a request for threats. Returns threat_score (0-1) + HMAC proof.",
)
async def detect_anomaly(request: AnomalyRequest):
    """
    **Frontend endpoint:** `POST /tee/detect`

    Run SGX anomaly detection on any request.
    Returns sealed result with unforgeable HMAC proof.
    """
    from services.anomaly import get_detector
    detector = get_detector()
    result = detector.detect(request.dict())
    return result


@router.post(
    "/verify",
    summary="Verify HMAC Proof",
    description="Verify a sealed result hasn't been tampered with.",
)
async def verify_proof(request: VerifyProofRequest):
    """
    **Frontend endpoint:** `POST /tee/verify`

    Check if a sealed result is authentic (proof matches data).
    Detects any tampering with threat scores.
    """
    from services.anomaly import AnomalyDetector
    sealed = {"model_output": request.model_output, "proof": request.proof}
    is_valid = AnomalyDetector.verify_proof(sealed)
    return {
        "valid": is_valid,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post(
    "/encrypt",
    summary="Encrypt Patient Vitals",
    description="Encrypt vitals using CKKS or AES. Returns encrypted data.",
)
async def encrypt_vitals(vitals: VitalsRequest):
    """
    **Frontend endpoint:** `POST /tee/encrypt`

    Encrypt patient vitals for secure transmission.
    Uses CKKS (if TenSEAL available) or AES-256.
    """
    from services.he_crypto import get_he_context
    he = get_he_context()

    vitals_dict = vitals.dict()
    result = he.encrypt_vitals(vitals_dict)
    result["vitals_received"] = list(vitals_dict.keys())

    return result


@router.post(
    "/decrypt",
    summary="Decrypt Encrypted Result",
    description="Decrypt an encrypted prediction result.",
)
async def decrypt_result(encrypted: EncryptedDataRequest):
    """
    **Frontend endpoint:** `POST /tee/decrypt`

    Decrypt an encrypted result from /tee/encrypted_predict.
    Only works with the correct encryption key.
    """
    from services.he_crypto import get_he_context
    he = get_he_context()

    enc_data = {
        "encrypted": encrypted.encrypted,
        "mode": encrypted.mode,
        "feature_names": encrypted.feature_names,
        "feature_count": encrypted.feature_count,
    }
    return he.decrypt_result(enc_data)


@router.post(
    "/encrypted_predict",
    summary="AI Prediction on Encrypted Data",
    description="Run AI risk prediction on encrypted vitals. Data stays encrypted.",
)
async def encrypted_predict(encrypted: EncryptedDataRequest):
    """
    **Frontend endpoint:** `POST /tee/encrypted_predict`

    Run AI prediction on encrypted patient vitals.
    With CKKS: AI runs on ciphertext (never decrypted).
    With AES: decrypts internally, predicts, re-encrypts.
    Result is encrypted — call /tee/decrypt to read.
    """
    from services.he_crypto import get_he_context
    he = get_he_context()

    enc_data = {
        "encrypted": encrypted.encrypted,
        "mode": encrypted.mode,
        "feature_names": encrypted.feature_names,
        "feature_count": encrypted.feature_count,
    }
    return he.encrypted_predict(enc_data)


@router.get(
    "/public_key",
    summary="Get HE Public Key",
    description="Get the public key for client-side encryption (base64).",
)
async def get_public_key():
    """
    **Frontend endpoint:** `GET /tee/public_key`

    Get the base64-encoded public key for encrypting vitals
    on the client side before sending to the server.
    """
    from services.he_crypto import get_he_context
    he = get_he_context()
    return {
        "public_key": he.get_public_key(),
        "mode": he.mode,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/health",
    summary="TEE Health Check",
    description="Quick health check for all TEE services.",
)
async def tee_health():
    """
    **Frontend endpoint:** `GET /tee/health`

    Quick health check. Returns status of all TEE components.
    """
    from services.anomaly import get_detector
    from services.he_crypto import get_he_context

    detector = get_detector()
    he = get_he_context()

    return {
        "status": "healthy",
        "anomaly_detection": {
            "initialized": detector.model is not None or detector.model_source == "rule-based",
            "model_source": detector.model_source,
            "detection_count": detector.detection_count,
        },
        "homomorphic_encryption": he.info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/status",
    summary="Detailed TEE Status",
    description="Detailed status of all TEE components including security info.",
)
async def tee_status(request: Request):
    """
    **Frontend endpoint:** `GET /tee/status`

    Detailed status of the TEE system.
    Includes threat assessment of the current request (from middleware).
    """
    from services.anomaly import get_detector
    from services.he_crypto import get_he_context
    from utils.proof import generate_nonce

    detector = get_detector()
    he = get_he_context()

    # Get middleware threat assessment (if available)
    threat_assessment = getattr(request.state, "threat_assessment", None)

    import os
    return {
        "system": {
            "tee_gateway": True,
            "anomaly_detection": True,
            "homomorphic_encryption": True,
            "block_threshold": float(os.getenv("TEE_BLOCK_THRESHOLD", "0.8")),
        },
        "anomaly": {
            "model_source": detector.model_source,
            "detection_count": detector.detection_count,
        },
        "encryption": he.info,
        "current_request": {
            "threat_assessment": threat_assessment,
        } if threat_assessment else None,
        "nonce": generate_nonce(),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }

# ═══════════════════════════════════════════════════════════════════
# MERKLE AUDIT TRAIL ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

class AuditLogRequest(BaseModel):
    event_type: str = Field(..., description="Type of event: PATIENT_READ, VITALS_UPDATE, PREDICTION, etc.")
    actor: str = Field(default="system", description="Who performed the action")
    data: Dict[str, Any] = Field(default_factory=dict, description="Event details")


@router.post(
    "/audit/log",
    summary="Log Event to Merkle Audit Trail",
    description="Add an immutable, tamper-proof entry to the Merkle tree audit trail.",
)
async def audit_log(request: AuditLogRequest):
    """
    **Frontend endpoint:** `POST /tee/audit/log`

    Log an event to the immutable Merkle tree audit trail.
    Returns leaf_hash + root_hash + Merkle proof.

    Even if a hacker deletes database entries, the Merkle root
    won't match → tampering detected.
    """
    from services.merkle_audit import get_merkle_tree
    tree = get_merkle_tree()
    result = tree.add_entry(
        event_type=request.event_type,
        data=request.data,
        actor=request.actor,
    )
    return result


@router.get(
    "/audit/root",
    summary="Get Current Merkle Root Hash",
    description="Get the current root hash of the audit Merkle tree.",
)
async def audit_root():
    """
    **Frontend endpoint:** `GET /tee/audit/root`

    The root hash is a fingerprint of the ENTIRE audit trail.
    If ANY entry is modified or deleted, the root hash changes.
    Frontend can cache this and compare periodically.
    """
    from services.merkle_audit import get_merkle_tree
    tree = get_merkle_tree()
    return {
        "root_hash": tree.root_hash,
        "entry_count": tree.entry_count,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post(
    "/audit/verify_integrity",
    summary="Verify Audit Trail Integrity",
    description="Rebuild the Merkle tree and verify no entries have been tampered with.",
)
async def audit_verify_integrity():
    """
    **Frontend endpoint:** `POST /tee/audit/verify_integrity`

    Verifies the ENTIRE audit trail is intact.
    Rebuilds the Merkle tree from stored entries and
    compares root hashes.

    If they differ → TAMPERING DETECTED.
    """
    from services.merkle_audit import get_merkle_tree
    tree = get_merkle_tree()
    return tree.verify_integrity()


class MerkleProofVerifyRequest(BaseModel):
    leaf_hash: str
    proof: List[Dict[str, str]]
    root_hash: str


@router.post(
    "/audit/verify_proof",
    summary="Verify Merkle Proof (ZKP-style)",
    description="Verify a specific log entry exists in the tree WITHOUT seeing other entries.",
)
async def audit_verify_proof(request: MerkleProofVerifyRequest):
    """
    **Frontend endpoint:** `POST /tee/audit/verify_proof`

    Zero-Knowledge-style verification: confirm a log entry
    is in the Merkle tree without revealing other entries.

    This is the "ZKP integrity proof" from your TEE description.
    """
    from services.merkle_audit import MerkleTree
    is_valid = MerkleTree.verify_proof(
        leaf_hash=request.leaf_hash,
        proof=request.proof,
        root_hash=request.root_hash,
    )
    return {
        "valid": is_valid,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get(
    "/audit/recent",
    summary="Get Recent Audit Entries",
    description="Get the last N audit log entries.",
)
async def audit_recent(count: int = 10):
    """
    **Frontend endpoint:** `GET /tee/audit/recent?count=10`

    Get recent entries from the Merkle audit trail.
    """
    from services.merkle_audit import get_merkle_tree
    tree = get_merkle_tree()
    entries = tree.get_recent_entries(min(count, 100))
    return {
        "entries": entries,
        "count": len(entries),
        "root_hash": tree.root_hash,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════
# ATTESTATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@router.get(
    "/attest",
    summary="Get Attestation Quote",
    description="Get a signed quote proving the TEE code hasn't been modified.",
)
async def attest():
    """
    **Frontend endpoint:** `GET /tee/attest`

    Simulated Remote Attestation — proves the TEE code
    hasn't been tampered with since deployment.

    In real SGX: this would be signed by the CPU's
    embedded key and verified by Intel's Attestation Service.

    Returns:
    - code_measurement: hash of all critical source files
    - code_intact: True if no files modified since boot
    - signature: HMAC proof (can't forge without key)
    """
    from services.attestation import get_attestation
    attestation = get_attestation()
    return attestation.get_quote()


class VerifyAttestationRequest(BaseModel):
    quote: Dict[str, Any]


@router.post(
    "/attest/verify",
    summary="Verify Attestation Quote",
    description="Verify a previously generated attestation quote.",
)
async def verify_attestation(request: VerifyAttestationRequest):
    """
    **Frontend endpoint:** `POST /tee/attest/verify`

    Verify an attestation quote to confirm:
    1. The signature is valid (not forged)
    2. The code hasn't been modified
    3. The TEE is running the expected code
    """
    from services.attestation import get_attestation
    attestation = get_attestation()
    return attestation.verify_quote(request.quote)


# ═══════════════════════════════════════════════════════════════════
# COMPREHENSIVE SECURITY STATUS
# ═══════════════════════════════════════════════════════════════════

@router.get(
    "/security_report",
    summary="Full TEE Security Report",
    description="Complete security status including attestation, Merkle integrity, and encryption.",
)
async def security_report():
    """
    **Frontend endpoint:** `GET /tee/security_report`

    Complete security report. Frontend should call this
    periodically and display in the security dashboard.

    Returns:
    - attestation: is code intact?
    - merkle_integrity: is audit trail intact?
    - encryption_mode: what encryption is active?
    - threat_model: honest assessment of what's protected
    """
    from services.anomaly import get_detector
    from services.he_crypto import get_he_context
    from services.merkle_audit import get_merkle_tree
    from services.attestation import get_attestation

    detector = get_detector()
    he = get_he_context()
    tree = get_merkle_tree()
    attestation = get_attestation()

    # Check attestation
    quote = attestation.get_quote()

    # Check Merkle integrity
    integrity = tree.verify_integrity()

    return {
        "overall_status": "SECURE" if (
            quote["code_intact"] and integrity["intact"]
        ) else "COMPROMISED",
        "attestation": {
            "code_intact": quote["code_intact"],
            "code_measurement": quote["code_measurement"],
            "tee_type": quote["tee_type"],
            "uptime_seconds": quote["uptime_seconds"],
        },
        "audit_trail": {
            "intact": integrity["intact"],
            "entry_count": tree.entry_count,
            "root_hash": tree.root_hash,
        },
        "encryption": he.info,
        "anomaly_detection": {
            "model_source": detector.model_source,
            "detection_count": detector.detection_count,
        },
        "threat_model": {
            "protects_against": [
                "Network eavesdropping (TLS + encryption)",
                "Database breach (encrypted data at rest)",
                "Data tampering (HMAC proof verification)",
                "Audit trail forgery (Merkle tree integrity)",
                "Unauthorized access (anomaly detection + gateway)",
                "Code modification detection (attestation)",
            ],
            "does_NOT_protect_against": [
                "Root-level OS compromise (need real SGX hardware)",
                "Memory dumping by privileged attacker",
                "Side-channel attacks (timing, power)",
                "Physical hardware tampering",
            ],
            "upgrade_path": [
                "Deploy on SGX-capable server for hardware TEE",
                "Use Gramine to run entire app in SGX enclave",
                "Add Intel Attestation Service for real quotes",
            ],
        },
        "reported_at": datetime.now(timezone.utc).isoformat(),
    }
# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE: Full Pipeline Endpoint
# ═══════════════════════════════════════════════════════════════════

@router.post(
    "/secure_vitals_pipeline",
    summary="Full Secure Vitals Pipeline",
    description="Encrypt → Predict → Return sealed result. One call does everything.",
)
async def secure_vitals_pipeline(vitals: VitalsRequest):
    """
    **Frontend endpoint:** `POST /tee/secure_vitals_pipeline`

    All-in-one endpoint:
    1. Encrypt vitals
    2. Run AI prediction on encrypted data
    3. Decrypt result
    4. Seal with HMAC proof
    5. Return sealed result

    This is the easiest endpoint for the frontend to use.
    """
    from services.anomaly import get_detector
    from services.he_crypto import get_he_context
    from utils.proof import seal_data

    he = get_he_context()
    detector = get_detector()

    # Step 1: Encrypt
    vitals_dict = vitals.dict()
    encrypted = he.encrypt_vitals(vitals_dict)

    # Step 2: Predict on encrypted data
    prediction_result = he.encrypted_predict(encrypted)

    # Step 3: Decrypt result
    if "encrypted_prediction" in prediction_result:
        decrypted = he.decrypt_result(prediction_result["encrypted_prediction"])
        prediction_data = decrypted.get("data", {})
    else:
        prediction_data = {"error": "Prediction failed"}

    # Step 4: Also run anomaly detection on the request itself
    anomaly_result = detector.detect({
        "endpoint": "/tee/secure_vitals_pipeline",
        "method": "POST",
        "user": "frontend_user",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_data": vitals_dict,
    })

    # Step 5: Seal everything
    combined_data = {
        "vitals_encrypted": True,
        "encryption_mode": encrypted["mode"],
        "prediction": prediction_data,
        "request_threat_score": anomaly_result["model_output"]["threat_score"],
        "request_threat_type": anomaly_result["model_output"]["threat_type"],
    }

    sealed = seal_data(combined_data)

    return {
        "sealed_result": sealed,
        "encryption_mode": encrypted["mode"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }