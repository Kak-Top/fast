# engine/models.py
from pydantic import BaseModel, Field
from typing import Dict, Optional

class RiskScoreResponse(BaseModel):
    patient_id: str
    sepsis_risk_score: float
    risk_level: str
    confidence: float
    compression_stats: Dict
    inference_metadata: Dict
    alert_color: str
    recommended_action: str
    timestamp: int