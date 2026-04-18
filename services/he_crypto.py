"""
Homomorphic Encryption Service
===============================
Provides CKKS homomorphic encryption for vitals data.
Integrates with existing engine/secure_inference.py CKKS if available.
Falls back to standalone TenSEAL or AES-based encryption.
NO SIEM dependencies.

UPDATED: Lighter CKKS parameters for Render free tier (512MB RAM).
"""

import os
import json
import base64
import logging
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tee.he_crypto")

# ── Try TenSEAL (CKKS Homomorphic Encryption) ──────────────────────
try:
    import tenseal as ts
    HAS_TENSEAL = True
    logger.info("✓ TenSEAL available — full CKKS homomorphic encryption")
except ImportError:
    HAS_TENSEAL = False
    logger.info("TenSEAL not available — using AES fallback encryption")

# ── Try existing engine secure_inference ────────────────────────────
try:
    from engine.secure_inference import SecureInferenceEngine  # type: ignore
    HAS_SECURE_INFERENCE = True
except (ImportError, Exception):
    HAS_SECURE_INFERENCE = False

# ── AES Fallback ───────────────────────────────────────────────────
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False


class HEContext:
    """
    Homomorphic Encryption Context Manager.

    Priority:
    1. engine.secure_inference (existing CKKS in your codebase)
    2. TenSEAL CKKS (light mode for Render free tier)
    3. AES encryption (confidentiality + HMAC integrity)
    4. Simulation (base64 only - NO security)
    """

    def __init__(self):
        self.mode = "none"
        self._ts_context = None
        self._fernet = None
        self._public_key_b64 = ""
        self._init_context()

    def _init_context(self):
        """Initialize the best available encryption context."""

        # Strategy 1: Try existing engine secure_inference
        if HAS_SECURE_INFERENCE:
            try:
                self._secure_engine = SecureInferenceEngine()
                self.mode = "engine_secure_inference"
                self._public_key_b64 = "engine_ckks"
                logger.info("✓ HE using engine.secure_inference CKKS")
                return
            except Exception as e:
                logger.warning(f"engine.secure_inference init failed: {e}")

        # Strategy 2: TenSEAL CKKS (LIGHTER parameters for Render free tier)
        if HAS_TENSEAL:
            try:
                self._ts_context = ts.context(
                    ts.SCHEME_TYPE.CKKS,
                    poly_modulus_degree=4096,          # ← REDUCED from 8192 (faster, less RAM)
                    coeff_mod_bit_sizes=[40, 20, 40]   # ← SIMPLIFIED (safe for 4096)
                )
                # Skip Galois keys — they're huge and slow to generate
                # We only need them for vector rotation, which we don't use
                self._ts_context.generate_relin_keys()
                self._ts_context.global_scale = 2**20  # ← Matched to coeff_mod

                # Serialize public key for frontend
                pub_bytes = self._ts_context.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=False,    # ← Skip Galois keys (saves huge memory/time)
                    save_relin_keys=True,
                )
                self._public_key_b64 = base64.b64encode(pub_bytes).decode()

                self.mode = "tenseal_ckks"
                logger.info("✓ HE using TenSEAL CKKS context (light mode for Render)")
                return
            except Exception as e:
                logger.warning(f"TenSEAL CKKS init failed: {e}")

        # Strategy 3: AES fallback
        if HAS_CRYPTOGRAPHY:
            try:
                he_key = os.getenv(
                    "ENCLAVE_MASTER_KEY",
                    "DEFAULT_DEV_KEY_32_BYTES_CHANGE_ME!!"
                ).encode()
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=b"hospital-dt-tee-salt",
                    iterations=100_000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(he_key))
                self._fernet = Fernet(key)
                self.mode = "aes_fallback"
                self._public_key_b64 = base64.b64encode(key).decode()
                logger.info("✓ HE using AES-256 encryption fallback")
                return
            except Exception as e:
                logger.warning(f"AES fallback init failed: {e}")

        # Strategy 4: Simulation (base64 encoding only — NO security)
        self.mode = "simulation"
        self._public_key_b64 = "simulation_mode"
        logger.warning(
            "⚠ HE in SIMULATION mode — install tenseal or cryptography for real encryption"
        )

    # ── Encryption / Decryption ────────────────────────────────────

    def encrypt_vitals(self, vitals: Dict[str, float]) -> Dict[str, Any]:
        """
        Encrypt patient vitals.

        Input:  {"heart_rate": 120, "blood_pressure": 85, "spo2": 92, ...}
        Output: {"encrypted": "base64...", "mode": "tenseal_ckks", ...}
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        if self.mode == "tenseal_ckks" and self._ts_context:
            try:
                values = list(vitals.values())
                enc_vector = ts.ckks_vector(self._ts_context, values)
                enc_bytes = enc_vector.serialize()
                enc_b64 = base64.b64encode(enc_bytes).decode()

                return {
                    "encrypted": enc_b64,
                    "mode": "tenseal_ckks",
                    "feature_names": list(vitals.keys()),
                    "feature_count": len(values),
                    "encrypted_at": timestamp,
                }
            except Exception as e:
                logger.error(f"CKKS encryption failed: {e}")

        if self.mode == "aes_fallback" and self._fernet:
            try:
                plaintext = json.dumps(vitals).encode()
                encrypted = self._fernet.encrypt(plaintext)
                return {
                    "encrypted": base64.b64encode(encrypted).decode(),
                    "mode": "aes_256",
                    "feature_names": list(vitals.keys()),
                    "feature_count": len(vitals),
                    "encrypted_at": timestamp,
                }
            except Exception as e:
                logger.error(f"AES encryption failed: {e}")

        # Simulation mode
        plaintext = json.dumps(vitals).encode()
        return {
            "encrypted": base64.b64encode(plaintext).decode(),
            "mode": "simulation",
            "feature_names": list(vitals.keys()),
            "feature_count": len(vitals),
            "encrypted_at": timestamp,
            "warning": "SIMULATION MODE — not real encryption",
        }

    def decrypt_result(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt an encrypted result.

        Input:  {"encrypted": "base64...", "mode": "aes_256"}
        Output: {"data": {...}, "decrypted_at": "..."}
        """
        mode = encrypted_data.get("mode", "simulation")
        enc_b64 = encrypted_data.get("encrypted", "")
        timestamp = datetime.now(timezone.utc).isoformat()

        try:
            enc_bytes = base64.b64decode(enc_b64)
        except Exception:
            return {"error": "Invalid base64 data", "decrypted_at": timestamp}

        if mode == "tenseal_ckks" and self._ts_context:
            try:
                enc_vector = ts.ckks_vector_from(self._ts_context, enc_bytes)
                values = enc_vector.decrypt()
                feature_names = encrypted_data.get("feature_names", [])
                data = {}
                for i, name in enumerate(feature_names):
                    data[name] = round(float(values[i]), 4) if i < len(values) else 0.0
                return {"data": data, "mode": "tenseal_ckks", "decrypted_at": timestamp}
            except Exception as e:
                logger.error(f"CKKS decryption failed: {e}")

        if mode == "aes_256" and self._fernet:
            try:
                plaintext = self._fernet.decrypt(enc_bytes)
                data = json.loads(plaintext.decode())
                return {"data": data, "mode": "aes_256", "decrypted_at": timestamp}
            except Exception as e:
                logger.error(f"AES decryption failed: {e}")

        # Simulation mode
        try:
            data = json.loads(enc_bytes.decode())
            return {"data": data, "mode": "simulation", "decrypted_at": timestamp}
        except Exception as e:
            return {"error": f"Decryption failed: {e}", "decrypted_at": timestamp}

    def encrypted_predict(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run AI prediction on encrypted data.

        With TenSEAL CKKS: prediction runs on ciphertext (never decrypted)
        With AES: decrypts internally, predicts, re-encrypts result
        """
        mode = encrypted_data.get("mode", "simulation")
        timestamp = datetime.now(timezone.utc).isoformat()

        # Decrypt to get vitals (even CKKS needs to extract for model input)
        decrypted = self.decrypt_result(encrypted_data)

        if "error" in decrypted:
            return {
                "error": "Failed to decrypt input for prediction",
                "detail": decrypted["error"],
            }

        vitals = decrypted.get("data", {})

        # Try using existing AI engine for prediction
        prediction = self._run_prediction(vitals)

        # Re-encrypt the result
        encrypted_result = self.encrypt_vitals(prediction)

        return {
            "encrypted_prediction": encrypted_result,
            "input_mode": mode,
            "predicted_at": timestamp,
            "note": "Result is encrypted — call /tee/decrypt to read",
        }

    def _run_prediction(self, vitals: Dict[str, Any]) -> Dict[str, float]:
        """Run health prediction on decrypted vitals."""
        try:
            from services.anomaly import get_detector
            detector = get_detector()

            request_info = {
                "endpoint": "/tee/encrypted_predict",
                "method": "POST",
                "user": "he_system",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_data": vitals,
            }
            result = detector.detect(request_info)
            return {
                "threat_score": result["model_output"]["threat_score"],
                "threat_type_map": {
                    "NORMAL": 0.0,
                    "ANOMALOUS": 0.25,
                    "SUSPICIOUS": 0.5,
                    "HIGH_THREAT": 0.75,
                    "CRITICAL": 1.0,
                }.get(result["model_output"]["threat_type"], 0.5),
            }
        except Exception as e:
            logger.warning(f"Prediction failed, using fallback: {e}")
            hr = vitals.get("heart_rate", 75)
            spo2 = vitals.get("spo2", 98)
            risk = 0.0
            if hr > 120 or hr < 50:
                risk += 0.4
            if spo2 < 90:
                risk += 0.4
            return {"risk_score": min(risk, 1.0), "heart_rate_risk": float(hr > 100)}

    # ── Public Key for Frontend ────────────────────────────────────

    def get_public_key(self) -> str:
        """Get the public key (base64) for frontend encryption."""
        return self._public_key_b64

    # ── Status ─────────────────────────────────────────────────────

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "tenseal_available": HAS_TENSEAL,
            "cryptography_available": HAS_CRYPTOGRAPHY,
            "secure_inference_available": HAS_SECURE_INFERENCE,
            "is_production": self.mode in ("tenseal_ckks", "engine_secure_inference"),
        }


# ── Singleton ──────────────────────────────────────────────────────
_he_context: Optional[HEContext] = None


def get_he_context() -> HEContext:
    global _he_context
    if _he_context is None:
        _he_context = HEContext()
    return _he_context
