"""
AI Models Router
----------------
Simulates ML model predictions. In production, replace the rule-based
logic with actual trained models:
  - Sepsis/deterioration risk  → XGBoost or LSTM (PyTorch)
  - Length of Stay prediction  → Gradient Boosting (scikit-learn)

Install real models with:
  pip install scikit-learn xgboost torch
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from dependencies import fake_patients_db, fake_vitals_db, get_current_user

router = APIRouter()


def _compute_risk_score(vitals: dict) -> dict:
    """
    Rule-based risk scoring (replace with XGBoost/LSTM in production).
    Returns a score 0-100 and risk category.
    """
    score = 0

    # Oxygen saturation (most critical for ICU)
    spo2 = vitals.get("spo2", 100)
    if spo2 < 90:   score += 35
    elif spo2 < 94: score += 20

    # Heart rate
    hr = vitals.get("heart_rate", 80)
    if hr > 120 or hr < 50: score += 20
    elif hr > 100:           score += 10

    # Blood pressure (systolic)
    sbp = vitals.get("blood_pressure_sys", 120)
    if sbp < 90:    score += 25
    elif sbp < 100: score += 15

    # Respiratory rate
    rr = vitals.get("respiratory_rate", 16)
    if rr > 25:    score += 15
    elif rr > 20:  score += 8

    # Temperature
    temp = vitals.get("temperature", 37)
    if temp > 39.0 or temp < 35.5: score += 5

    score = min(score, 100)

    if score >= 70:   category = "HIGH RISK"
    elif score >= 40: category = "MODERATE RISK"
    else:             category = "LOW RISK"

    return {"score": score, "category": category}


def _predict_los(vitals: dict, age: int, diagnosis: str) -> dict:
    """
    Simulated Length of Stay prediction.
    Replace with Gradient Boosting regressor in production.
    """
    base_days = 3
    if vitals.get("spo2", 100) < 92:   base_days += 4
    if vitals.get("heart_rate", 80) > 110: base_days += 2
    if age > 65:                         base_days += 2
    if "sepsis" in diagnosis.lower():    base_days += 5
    if "respiratory" in diagnosis.lower(): base_days += 3

    return {
        "predicted_days": base_days,
        "confidence": "72%",
        "model": "GradientBoostingRegressor (simulated)",
    }


# ── GET /icu/ai/risk/{patient_id} ─────────────────────────────────────────────
@router.get("/risk/{patient_id}", summary="Get AI sepsis & deterioration risk score")
def get_risk_score(patient_id: str, current_user=Depends(get_current_user)):
    """
    Runs the AI risk model on the patient's latest vitals.
    Flags sepsis risk, deterioration probability, and recommended actions.

    **Production model:** XGBoost classifier or LSTM on time-series vitals.

    **Sample Response:**
    ```json
    {
      "patient_id": "P001",
      "name": "Khalid Al-Mansouri",
      "diagnosis": "Respiratory Failure",
      "risk_assessment": {
        "overall_score": 75,
        "category": "HIGH RISK",
        "sepsis_probability": "68%",
        "deterioration_probability": "71%"
      },
      "contributing_factors": [
        "SpO2 critically low: 89%",
        "Respiratory rate elevated: 28 bpm",
        "Systolic BP low: 85 mmHg"
      ],
      "recommended_actions": [
        "Increase oxygen therapy",
        "Notify attending physician immediately",
        "Consider vasopressor therapy"
      ],
      "model_info": {
        "model": "XGBoost (simulated)",
        "last_updated": "2025-03-02T08:15:00"
      }
    }
    ```
    """
    patient = fake_patients_db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    history = fake_vitals_db.get(patient_id, [])
    if not history:
        raise HTTPException(status_code=400, detail="No vitals data available for this patient")

    latest = history[-1]
    risk = _compute_risk_score(latest)

    # Build contributing factors
    factors = []
    if latest.get("spo2", 100) < 94:
        factors.append(f"SpO2 {'critically' if latest['spo2'] < 90 else ''} low: {latest['spo2']}%")
    if latest.get("respiratory_rate", 16) > 20:
        factors.append(f"Respiratory rate elevated: {latest['respiratory_rate']} bpm")
    if latest.get("blood_pressure_sys", 120) < 100:
        factors.append(f"Systolic BP low: {latest['blood_pressure_sys']} mmHg")
    if latest.get("heart_rate", 80) > 100:
        factors.append(f"Tachycardia: {latest['heart_rate']} bpm")
    if latest.get("temperature", 37) > 38.5:
        factors.append(f"Fever: {latest['temperature']}°C")

    # Recommendations based on risk
    actions = []
    if risk["score"] >= 70:
        actions = ["Notify attending physician immediately", "Increase monitoring frequency", "Prepare for possible escalation of care"]
        if latest.get("spo2", 100) < 92:
            actions.append("Increase oxygen therapy")
        if latest.get("blood_pressure_sys", 120) < 90:
            actions.append("Consider vasopressor therapy")
    elif risk["score"] >= 40:
        actions = ["Increase monitoring to every 15 minutes", "Review medication orders", "Notify charge nurse"]
    else:
        actions = ["Continue routine monitoring", "No immediate action required"]

    sepsis_prob = min(risk["score"] + 8, 100)
    deterioration_prob = min(risk["score"] + 3, 100)

    return {
        "patient_id": patient_id,
        "name": patient["name"],
        "diagnosis": patient["diagnosis"],
        "assessed_at": datetime.utcnow().isoformat(),
        "risk_assessment": {
            "overall_score": risk["score"],
            "category": risk["category"],
            "sepsis_probability": f"{sepsis_prob}%",
            "deterioration_probability": f"{deterioration_prob}%",
        },
        "contributing_factors": factors if factors else ["All vitals within acceptable range"],
        "recommended_actions": actions,
        "model_info": {
            "model": "XGBoost (simulated — replace with trained model)",
            "inputs_used": "SpO2, HR, BP, RR, Temperature",
            "last_vitals_at": latest["timestamp"],
        },
    }


# ── GET /icu/ai/predict/los/{patient_id} ──────────────────────────────────────
@router.get("/predict/los/{patient_id}", summary="Predict patient Length of Stay")
def predict_los(patient_id: str, current_user=Depends(get_current_user)):
    """
    Predicts how many more days the patient is likely to remain in the ICU.

    **Production model:** Gradient Boosting Regressor (scikit-learn).

    **Sample Response:**
    ```json
    {
      "patient_id": "P001",
      "name": "Khalid Al-Mansouri",
      "predicted_los": {
        "predicted_days": 9,
        "confidence": "72%",
        "model": "GradientBoostingRegressor (simulated)"
      },
      "key_factors": ["Age > 65", "SpO2 < 92%", "Diagnosis: Respiratory Failure"]
    }
    ```
    """
    patient = fake_patients_db.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    history = fake_vitals_db.get(patient_id, [])
    latest = history[-1] if history else {}

    los = _predict_los(latest, patient["age"], patient["diagnosis"])

    key_factors = []
    if patient["age"] > 65: key_factors.append("Age > 65")
    if latest.get("spo2", 100) < 92: key_factors.append("SpO2 < 92%")
    if "sepsis" in patient["diagnosis"].lower(): key_factors.append("Diagnosis: Sepsis")
    if "respiratory" in patient["diagnosis"].lower(): key_factors.append("Diagnosis: Respiratory Failure")

    return {
        "patient_id": patient_id,
        "name": patient["name"],
        "diagnosis": patient["diagnosis"],
        "predicted_los": los,
        "key_factors": key_factors if key_factors else ["Standard case — no major risk factors detected"],
    }


# ── GET /icu/ai/alerts ─────────────────────────────────────────────────────────
@router.get("/alerts", summary="Get all active AI-generated risk alerts")
def get_ai_alerts(current_user=Depends(get_current_user)):
    """
    Scans all ICU patients and returns those flagged by the AI risk model.

    **Sample Response:**
    ```json
    {
      "total_alerts": 1,
      "alerts": [
        {
          "patient_id": "P001",
          "name": "Khalid Al-Mansouri",
          "bed_id": "ICU-01",
          "risk_score": 75,
          "category": "HIGH RISK",
          "top_factor": "SpO2 critically low: 89%",
          "generated_at": "2025-03-02T09:00:00"
        }
      ]
    }
    ```
    """
    alerts = []
    for pid, patient in fake_patients_db.items():
        history = fake_vitals_db.get(pid, [])
        if not history:
            continue
        latest = history[-1]
        risk = _compute_risk_score(latest)
        if risk["score"] >= 40:
            factors = []
            if latest.get("spo2", 100) < 94:
                factors.append(f"SpO2 low: {latest['spo2']}%")
            if latest.get("heart_rate", 80) > 100:
                factors.append(f"Tachycardia: {latest['heart_rate']} bpm")
            if latest.get("blood_pressure_sys", 120) < 100:
                factors.append(f"Low BP: {latest['blood_pressure_sys']} mmHg")

            alerts.append({
                "patient_id": pid,
                "name": patient["name"],
                "bed_id": patient["bed_id"],
                "risk_score": risk["score"],
                "category": risk["category"],
                "top_factor": factors[0] if factors else "Multiple parameters out of range",
                "generated_at": datetime.utcnow().isoformat(),
            })

    alerts.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"total_alerts": len(alerts), "alerts": alerts}
