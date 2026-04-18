"""
SGX Anomaly Detection Service
==============================
Runs threat detection on every request using ML model + rule-based fallback.
Seals results with HMAC proof (unforgeable).
Compatible with existing engine/secure_inference.py CKKS setup.
NO SIEM dependencies.
"""

import hmac
import hashlib
import json
import os
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("tee.anomaly")

# Shared Enclave Master Key
EMK = os.getenv(
    "ENCLAVE_MASTER_KEY",
    "DEFAULT_DEV_KEY_32_BYTES_CHANGE_ME!!"
).encode()

# ── Check for sklearn (optional, for ML model) ──────────────────────
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.info("scikit-learn not available — using rule-based detection")

# ── Check for existing AI engine (optional integration) ──────────────
try:
    from engine.secure_inference import SecureInferenceEngine  # type: ignore
    HAS_SECURE_INFERENCE = True
    logger.info("✓ engine.secure_inference available — will integrate")
except (ImportError, Exception):
    HAS_SECURE_INFERENCE = False


class FeatureExtractor:
    """
    Extract 10 features from an HTTP request for anomaly detection.

    Features MUST be normalized to [0, 1] range.
    If you trained your model on different features, modify this class.
    """

    @staticmethod
    def extract(request_info: Dict[str, Any]) -> List[float]:
        endpoint = request_info.get("endpoint", "")
        method = request_info.get("method", "GET")
        user = request_info.get("user", "unknown")
        timestamp_str = request_info.get("timestamp", "")
        request_data = request_info.get("request_data", {})
        source_ip = request_info.get("source_ip", "")

        features: List[float] = []

        # 1. Endpoint complexity (path segments / 5)
        segments = len([x for x in endpoint.split('/') if x])
        features.append(min(segments / 5.0, 1.0))

        # 2. Request data size (KB normalized by 100)
        data_size = len(json.dumps(request_data)) / 1000.0
        features.append(min(data_size / 100.0, 1.0))

        # 3. Hour of day (0–23 → 0–1)
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            features.append(dt.hour / 24.0)
        except Exception:
            features.append(0.5)

        # 4. Is patient/medical endpoint?
        is_medical = 1.0 if any(
            kw in endpoint.lower()
            for kw in ['patient', 'vitals', 'icu', 'medical', 'ehr']
        ) else 0.0
        features.append(is_medical)

        # 5. Method risk (POST=1.0, DELETE=0.9, PUT=0.8, PATCH=0.7, GET=0.0)
        method_risk = {"POST": 1.0, "DELETE": 0.9, "PUT": 0.8,
                       "PATCH": 0.7, "GET": 0.0}.get(method.upper(), 0.5)
        features.append(method_risk)

        # 6. Is after hours? (before 6 AM or after 10 PM)
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            features.append(1.0 if dt.hour < 6 or dt.hour > 22 else 0.0)
        except Exception:
            features.append(0.0)

        # 7. Suspicious user?
        suspicious = {"admin", "root", "test", "unknown", "guest", "anonymous"}
        features.append(1.0 if user.lower() in suspicious else 0.0)

        # 8. External IP? (internal = 192.168.x / 10.x / 172.16-31.x)
        if source_ip:
            is_internal = source_ip.startswith(("192.168.", "10.", "172."))
            features.append(0.0 if is_internal else 0.6)
        else:
            features.append(0.3)

        # 9. Data sensitivity score
        sensitivity_map = [
            (['patient', 'icu', 'vitals', 'medical', 'ehr'], 0.9),
            (['audit', 'log', 'admin', 'config'], 0.8),
            (['auth', 'login', 'token', 'password'], 0.7),
            (['health', 'status', 'ping'], 0.1),
        ]
        sensitivity = 0.3  # default
        for keywords, score in sensitivity_map:
            if any(kw in endpoint.lower() for kw in keywords):
                sensitivity = score
                break
        features.append(sensitivity)

        # 10. Request pattern anomaly (stub — in production, track per user)
        features.append(0.1)

        assert len(features) == 10, f"Expected 10 features, got {len(features)}"
        return features


class AnomalyDetector:
    """
    SGX Anomaly Detection Engine.

    Model loading priority:
    1. ai_model.pkl file (your trained 85% model)
    2. engine.secure_inference integration (existing CKKS setup)
    3. Auto-trained lightweight RandomForest (startup)
    4. Rule-based scoring (always works)
    """

    def __init__(self):
        self.model = None
        self.model_source = "none"
        self._detection_count = 0
        self._load_model()

    def _load_model(self):
        """Try multiple model loading strategies."""
        # Strategy 1: Load from ai_model.pkl
        model_path = os.getenv("MODEL_PATH", "models/ai_model.pkl")
        if os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_source = f"pkl:{model_path}"
                logger.info(f"✓ ML model loaded from {model_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")

        # Strategy 2: Try existing secure_inference engine
        if HAS_SECURE_INFERENCE:
            try:
                self.model = SecureInferenceEngine()
                self.model_source = "engine.secure_inference"
                logger.info("✓ Using engine.secure_inference for predictions")
                return
            except Exception as e:
                logger.warning(f"Failed to init secure_inference: {e}")

        # Strategy 3: Train lightweight model at startup
        if HAS_SKLEARN:
            try:
                self.model = self._train_default_model()
                self.model_source = "startup-trained"
                logger.info("✓ Lightweight RandomForest trained at startup")
                return
            except Exception as e:
                logger.warning(f"Failed to train startup model: {e}")

        # Strategy 4: Rule-based fallback
        self.model = None
        self.model_source = "rule-based"
        logger.info("✓ Using rule-based anomaly detection (no ML model)")

    def _train_default_model(self):
        """Train a lightweight RandomForest on synthetic data."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=7,
            n_redundant=2,
            n_classes=2,
            random_state=42,
            weights=[0.85, 0.15],
        )
        model = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=42
        )
        model.fit(X, y)
        return model

    # ── Main Detection Method ──────────────────────────────────────

    def detect(self, request_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run anomaly detection on a request. Returns sealed result.

        Input:  {"endpoint": "/icu/patients/P001", "method": "GET", ...}
        Output: {"model_output": {...}, "proof": "abc123...", "sealed_at": "..."}
        """
        features = FeatureExtractor.extract(request_info)
        threat_score = self._compute_threat_score(features)

        # Classify
        if threat_score > 0.8:
            threat_type = "CRITICAL"
        elif threat_score > 0.6:
            threat_type = "HIGH_THREAT"
        elif threat_score > 0.4:
            threat_type = "SUSPICIOUS"
        elif threat_score > 0.2:
            threat_type = "ANOMALOUS"
        else:
            threat_type = "NORMAL"

        model_output = {
            "endpoint": request_info.get("endpoint"),
            "method": request_info.get("method"),
            "user": request_info.get("user"),
            "threat_score": round(threat_score, 4),
            "threat_type": threat_type,
            "features": [round(f, 4) for f in features],
            "model_source": self.model_source,
            "detection_at": datetime.now(timezone.utc).isoformat(),
        }

        # HMAC seal (unforgeable proof)
        canonical = json.dumps(model_output, sort_keys=True, default=str)
        proof = hmac.new(EMK, canonical.encode(), hashlib.sha256).hexdigest()

        self._detection_count += 1

        return {
            "model_output": model_output,
            "proof": proof,
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        }

    def _compute_threat_score(self, features: List[float]) -> float:
        """Compute threat score using model or rules."""
        if self.model is not None:
            try:
                proba = self.model.predict_proba([features])
                threat_score = float(proba[0][1])  # probability of anomaly
                return max(0.0, min(1.0, threat_score))
            except AttributeError:
                try:
                    pred = self.model.predict([features])
                    return max(0.0, min(1.0, float(pred[0])))
                except Exception:
                    pass
            except Exception:
                pass

        return self._rule_based_score(features)

    @staticmethod
    def _rule_based_score(features: List[float]) -> float:
        """Rule-based anomaly scoring (no ML model needed)."""
        score = 0.0
        # After hours (feature 5)
        if features[5] > 0.5:
            score += 0.25
        # Suspicious user (feature 6)
        if features[6] > 0.5:
            score += 0.20
        # High method risk (feature 4)
        if features[4] > 0.7:
            score += 0.15
        # Large data (feature 1)
        if features[1] > 0.5:
            score += 0.15
        # High sensitivity (feature 8)
        if features[8] > 0.7:
            score += 0.10
        # External IP (feature 7)
        if features[7] > 0.3:
            score += 0.10
        # Complex endpoint (feature 0)
        if features[0] > 0.6:
            score += 0.05
        return min(score, 1.0)

    # ── Proof Verification ─────────────────────────────────────────

    @staticmethod
    def verify_proof(sealed_result: Dict[str, Any]) -> bool:
        """Verify HMAC proof of a sealed result (detects tampering)."""
        model_output = sealed_result.get("model_output", {})
        received_proof = sealed_result.get("proof", "")
        if not model_output or not received_proof:
            return False
        canonical = json.dumps(model_output, sort_keys=True, default=str)
        expected_proof = hmac.new(
            EMK, canonical.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(received_proof, expected_proof)

    # ── Status ─────────────────────────────────────────────────────

    @property
    def detection_count(self) -> int:
        return self._detection_count


# ── Singleton ──────────────────────────────────────────────────────
_detector: Optional[AnomalyDetector] = None


def get_detector() -> AnomalyDetector:
    global _detector
    if _detector is None:
        _detector = AnomalyDetector()
    return _detector