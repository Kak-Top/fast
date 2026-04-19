# engine/model_registry.py
"""
ModelRegistry
─────────────
Persists trained custom models IN-MEMORY (fast) and optionally to disk.

When a developer trains a model via POST /icu/ai/models/custom/train,
the registry stores:
  • The trained sklearn estimator (pickled)
  • Training metadata (accuracy, features, timestamp)
  • TurboQuant compression stats from the training run
  • CKKS capability flag

The frontend queries GET /icu/ai/models/custom/status to discover
which model is active and display its capabilities.

Thread-safety: asyncio single-thread — no locks needed.
"""
from __future__ import annotations

import pickle
import time
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class TrainedModelMeta:
    """All metadata the frontend needs to display about a trained model."""

    # Identity
    model_id: str
    model_name: str          # e.g. "RandomForest", "GradientBoosting", "XGBoost"
    version: int             # increments on each retrain

    # Training quality
    accuracy: float          # 0-1
    f1_score: float
    auc_roc: float
    n_samples: int
    n_features: int
    feature_names: List[str]

    # TurboQuant / CKKS
    turboquant_enabled: bool
    ckks_enabled: bool
    compression_ratio: str   # e.g. "5.3x"
    vram_saved_percent: float
    encoding_latency_ms: float

    # Status
    trained_at: float        # unix timestamp
    trained_by: str          # username
    is_active: bool = True
    description: str = ""

    # Serialised estimator (not sent to frontend)
    _estimator_pickle: bytes = field(default=b"", repr=False)

    def to_status_dict(self) -> Dict:
        """Frontend-facing dict — no pickle bytes."""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "is_active": self.is_active,
            "trained_at": self.trained_at,
            "trained_at_iso": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.trained_at)
            ),
            "trained_by": self.trained_by,
            "description": self.description,
            "performance": {
                "accuracy": round(self.accuracy, 4),
                "f1_score": round(self.f1_score, 4),
                "auc_roc": round(self.auc_roc, 4),
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "feature_names": self.feature_names,
            },
            "security": {
                "turboquant_enabled": self.turboquant_enabled,
                "ckks_enabled": self.ckks_enabled,
                "compression_ratio": self.compression_ratio,
                "vram_saved_percent": self.vram_saved_percent,
                "encoding_latency_ms": self.encoding_latency_ms,
                "inference_mode": (
                    "CKKS Homomorphic + 3-bit Polar"
                    if self.ckks_enabled
                    else "Standard"
                ),
            },
            "badge": _build_badge(self),
        }


def _build_badge(m: TrainedModelMeta) -> Dict:
    """Compute the UI badge shown in the frontend."""
    stars = "★" * min(5, round(m.accuracy * 5))
    if m.accuracy >= 0.90:
        tier, color = "ELITE", "#00ffc8"
    elif m.accuracy >= 0.80:
        tier, color = "STRONG", "#818cf8"
    elif m.accuracy >= 0.70:
        tier, color = "GOOD", "#f59e0b"
    else:
        tier, color = "BASELINE", "#64748b"

    return {
        "tier": tier,
        "color": color,
        "stars": stars,
        "label": f"{tier} · {m.accuracy*100:.1f}% accuracy",
        "secure_label": (
            f"🔐 {m.compression_ratio} · CKKS · {m.encoding_latency_ms:.1f}ms enc"
            if m.ckks_enabled
            else "Standard inference"
        ),
    }


class ModelRegistry:
    """Singleton model registry — one active model at a time per slot."""

    def __init__(self):
        self._active: Optional[TrainedModelMeta] = None
        self._history: List[TrainedModelMeta] = []
        self._version = 0

    # ── Write ─────────────────────────────────────────────────────────────────

    def register(self, meta: TrainedModelMeta) -> TrainedModelMeta:
        """Deactivate old model, register new one as active."""
        if self._active:
            self._active.is_active = False
            self._history.append(self._active)
        self._active = meta
        self._version += 1
        logger.info(
            f"✅ Model registered: {meta.model_name} v{meta.version} "
            f"acc={meta.accuracy:.3f} tq={meta.turboquant_enabled}"
        )
        return meta

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_active(self) -> Optional[TrainedModelMeta]:
        return self._active

    def get_estimator(self) -> Optional[Any]:
        if self._active and self._active._estimator_pickle:
            return pickle.loads(self._active._estimator_pickle)
        return None

    def has_active(self) -> bool:
        return self._active is not None

    def status(self) -> Dict:
        if not self._active:
            return {
                "has_model": False,
                "message": "No model trained yet. POST /icu/ai/models/custom/train to train one.",
            }
        return {
            "has_model": True,
            **self._active.to_status_dict(),
            "history_count": len(self._history),
        }

    def next_version(self) -> int:
        return self._version + 1

    def make_model_id(self, name: str) -> str:
        raw = f"{name}:{time.time()}"
        return "mdl_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


# ── Global singleton ──────────────────────────────────────────────────────────
registry = ModelRegistry()
