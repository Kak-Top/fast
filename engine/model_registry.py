# engine/model_registry.py
"""
Model Registry — Persistent Version
Stores trained models in the `trained_models` database table.
"""

from __future__ import annotations
import pickle
import time
import logging
from typing import Optional, Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update

logger = logging.getLogger(__name__)


class TrainedModelMeta:
    """Dataclass-like object representing a trained model's metadata."""
    
    def __init__(
        self,
        model_id: str,
        model_name: str,
        version: int,
        accuracy: float,
        f1_score: float = 0.0,
        auc_roc: float = 0.0,
        n_samples: int = 0,
        n_features: int = 8,
        feature_names: List[str] = None,
        turboquant_enabled: bool = False,
        ckks_enabled: bool = False,
        compression_ratio: str = "1.0x",
        vram_saved_percent: float = 0.0,
        encoding_latency_ms: float = 0.0,
        trained_at: float = 0.0,
        trained_by: str = "",
        description: str = "",
        _estimator_pickle: bytes = None,
        _db_id: int = None,  # Internal: database row ID
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.version = version
        self.accuracy = accuracy
        self.f1_score = f1_score
        self.auc_roc = auc_roc
        self.n_samples = n_samples
        self.n_features = n_features
        self.feature_names = feature_names or []
        self.turboquant_enabled = turboquant_enabled
        self.ckks_enabled = ckks_enabled
        self.compression_ratio = compression_ratio
        self.vram_saved_percent = vram_saved_percent
        self.encoding_latency_ms = encoding_latency_ms
        self.trained_at = trained_at
        self.trained_by = trained_by
        self.description = description
        self._estimator_pickle = _estimator_pickle
        self._db_id = _db_id

    def accuracy_tier(self) -> str:
        if self.accuracy >= 0.95:
            return "S"
        elif self.accuracy >= 0.90:
            return "A"
        elif self.accuracy >= 0.85:
            return "B"
        elif self.accuracy >= 0.80:
            return "C"
        return "D"

    def tier_stars(self) -> int:
        tiers = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
        return tiers.get(self.accuracy_tier(), 1)

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "accuracy": self.accuracy,
            "accuracy_tier": self.accuracy_tier(),
            "tier_stars": self.tier_stars(),
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "turboquant_enabled": self.turboquant_enabled,
            "ckks_enabled": self.ckks_enabled,
            "compression_ratio": self.compression_ratio,
            "vram_saved_percent": self.vram_saved_percent,
            "encoding_latency_ms": self.encoding_latency_ms,
            "trained_at": self.trained_at,
            "trained_by": self.trained_by,
            "description": self.description,
        }


class ModelRegistry:
    """
    Persistent model registry.
    
    - On startup: loads the active model from DB into memory (fast access)
    - On train: saves to DB + updates in-memory cache
    - On delete: marks as inactive in DB + clears memory
    """
    
    def __init__(self):
        self._active: Optional[TrainedModelMeta] = None
        self._loaded = False

    def make_model_id(self, model_name: str) -> str:
        """Generate a unique model ID."""
        import hashlib
        ts = str(int(time.time()))
        raw = f"{model_name}-{ts}"
        return f"custom-{hashlib.sha256(raw.encode()).hexdigest()[:12]}"

    def next_version(self) -> int:
        """Always 1 for now (we only keep one active model)."""
        return 1

    def has_active(self) -> bool:
        return self._active is not None

    def get_active(self) -> Optional[TrainedModelMeta]:
        return self._active

    def get_estimator(self) -> Optional[Dict[str, Any]]:
        """Get the unpickled estimator bundle."""
        if self._active is None or self._active._estimator_pickle is None:
            return None
        try:
            return pickle.loads(self._active._estimator_pickle)
        except Exception as e:
            logger.error(f"Failed to unpickle estimator: {e}")
            return None

    def register(self, meta: TrainedModelMeta) -> None:
        """Set the active model in memory. DB save happens in the endpoint."""
        self._active = meta
        logger.info(f"Model registered: {meta.model_id} (acc={meta.accuracy:.3f})")

    async def persist_to_db(self, meta: TrainedModelMeta, db: AsyncSession) -> None:
        """
        Save model to database.
        - Deactivates any previously active model
        - Inserts new model as active
        """
        from models import TrainedModel  # Import here to avoid circular imports

        # Deactivate all existing models
        try:
            await db.execute(
                update(TrainedModel).values(is_active=False)
            )
        except Exception as e:
            logger.warning(f"Could not deactivate old models: {e}")

        # Insert new model
        db_model = TrainedModel(
            model_id=meta.model_id,
            model_name=meta.model_name,
            version=meta.version,
            accuracy=meta.accuracy,
            f1_score=meta.f1_score,
            auc_roc=meta.auc_roc,
            n_samples=meta.n_samples,
            n_features=meta.n_features,
            feature_names=meta.feature_names,
            turboquant_enabled=meta.turboquant_enabled,
            ckks_enabled=meta.ckks_enabled,
            compression_ratio=meta.compression_ratio,
            vram_saved_percent=meta.vram_saved_percent,
            encoding_latency_ms=meta.encoding_latency_ms,
            trained_at=meta.trained_at,
            trained_by=meta.trained_by,
            description=meta.description,
            is_active=True,
            estimator_pickle=meta._estimator_pickle,
        )
        db.add(db_model)
        await db.commit()
        logger.info(f"Model persisted to DB: {meta.model_id}")

    async def load_from_db(self, db: AsyncSession) -> bool:
        """
        Load the active model from database into memory.
        Call this on app startup.
        """
        if self._loaded:
            return self._active is not None

        from models import TrainedModel

        try:
            result = await db.execute(
                select(TrainedModel).where(TrainedModel.is_active == True).limit(1)
            )
            db_model = result.scalar_one_or_none()

            if db_model and db_model.estimator_pickle:
                self._active = TrainedModelMeta(
                    model_id=db_model.model_id,
                    model_name=db_model.model_name,
                    version=db_model.version,
                    accuracy=db_model.accuracy,
                    f1_score=db_model.f1_score,
                    auc_roc=db_model.auc_roc,
                    n_samples=db_model.n_samples,
                    n_features=db_model.n_features,
                    feature_names=db_model.feature_names or [],
                    turboquant_enabled=db_model.turboquant_enabled,
                    ckks_enabled=db_model.ckks_enabled,
                    compression_ratio=db_model.compression_ratio,
                    vram_saved_percent=db_model.vram_saved_percent,
                    encoding_latency_ms=db_model.encoding_latency_ms,
                    trained_at=db_model.trained_at,
                    trained_by=db_model.trained_by,
                    description=db_model.description or "",
                    _estimator_pickle=db_model.estimator_pickle,
                    _db_id=db_model.id,
                )
                logger.info(f"Loaded model from DB: {db_model.model_id}")
            else:
                logger.info("No active model found in database")
        except Exception as e:
            logger.error(f"Failed to load model from DB: {e}")

        self._loaded = True
        return self._active is not None

    async def delete_from_db(self, db: AsyncSession) -> bool:
        """Deactivate the current model in DB."""
        from models import TrainedModel

        if self._active is None:
            return False

        try:
            await db.execute(
                update(TrainedModel)
                .where(TrainedModel.model_id == self._active.model_id)
                .values(is_active=False)
            )
            await db.commit()
            logger.info(f"Model deactivated in DB: {self._active.model_id}")
        except Exception as e:
            logger.error(f"Failed to deactivate model in DB: {e}")

        self._active = None
        return True

    def status(self) -> Dict[str, Any]:
        """Return status dict for the /status endpoint."""
        if self._active is None:
            return {
                "has_active_model": False,
                "message": "No model trained yet. Call POST /icu/ai/models/custom/train",
            }
        return {
            "has_active_model": True,
            "model": self._active.to_status_dict(),
        }


# Singleton instance
registry = ModelRegistry()
