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
from dependencies import get_current_user
from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from models import Patient, Vital
from sqlalchemy.future import select

router = APIRouter()


def _compute_risk_score(vitals: dict) -> dict:
    """
    Rule-based risk scoring (replace with XGBoost/LSTM in production).
    Returns a score 0-100 and risk category.
    """
    score = 0

    # Oxygen saturation (most critical for ICU)
    spo2 = vitals.get("spo2", 100)
    if spo2 is not None:
        if spo2 < 90:   score += 35
        elif spo2 < 94: score += 20

    # Heart rate
    hr = vitals.get("heart_rate", 80)
    if hr is not None:
        if hr > 120 or hr < 50: score += 20
        elif hr > 100:           score += 10

    # Blood pressure (systolic)
    sbp = vitals.get("blood_pressure_sys", 120)
    if sbp is not None:
        if sbp < 90:    score += 25
        elif sbp < 100: score += 15

    # Respiratory rate
    rr = vitals.get("respiratory_rate", 16)
    if rr is not None:
        if rr > 25:    score += 15
        elif rr > 20:  score += 8

    # Temperature
    temp = vitals.get("temperature", 37)
    if temp is not None:
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
    if vitals.get("spo2", 100) is not None and vitals.get("spo2", 100) < 92:   base_days += 4
    if vitals.get("heart_rate", 80) is not None and vitals.get("heart_rate", 80) > 110: base_days += 2
    if age and age > 65:                         base_days += 2
    if diagnosis and "sepsis" in diagnosis.lower():    base_days += 5
    if diagnosis and "respiratory" in diagnosis.lower(): base_days += 3

    return {
        "predicted_days": base_days,
        "confidence": "72%",
        "model": "GradientBoostingRegressor (simulated)",
    }


# ── GET /icu/ai/risk/{patient_id} ─────────────────────────────────────────────
@router.get("/risk/{patient_id}", summary="Get AI sepsis & deterioration risk score")
async def get_risk_score(patient_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Runs the AI risk model on the patient's latest vitals.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    v_query = await db.execute(select(Vital).where(Vital.patient_id == patient_id).order_by(Vital.timestamp.desc()).limit(1))
    latest = v_query.scalar_one_or_none()

    if not latest:
        raise HTTPException(status_code=400, detail="No vitals data available for this patient")

    latest_dict = {
        "heart_rate": latest.heart_rate,
        "blood_pressure_sys": latest.blood_pressure_sys,
        "blood_pressure_dia": latest.blood_pressure_dia,
        "spo2": latest.spo2,
        "respiratory_rate": latest.respiratory_rate,
        "temperature": latest.temperature,
        "timestamp": latest.timestamp.isoformat() if latest.timestamp else None
    }
    
    risk = _compute_risk_score(latest_dict)

    # Build contributing factors
    factors = []
    if latest_dict.get("spo2", 100) < 94:
        factors.append(f"SpO2 {'critically' if latest_dict['spo2'] < 90 else ''} low: {latest_dict['spo2']}%")
    if latest_dict.get("respiratory_rate", 16) > 20:
        factors.append(f"Respiratory rate elevated: {latest_dict['respiratory_rate']} bpm")
    if latest_dict.get("blood_pressure_sys", 120) < 100:
        factors.append(f"Systolic BP low: {latest_dict['blood_pressure_sys']} mmHg")
    if latest_dict.get("heart_rate", 80) > 100:
        factors.append(f"Tachycardia: {latest_dict['heart_rate']} bpm")
    if latest_dict.get("temperature", 37) > 38.5:
        factors.append(f"Fever: {latest_dict['temperature']}°C")

    # Recommendations based on risk
    actions = []
    if risk["score"] >= 70:
        actions = ["Notify attending physician immediately", "Increase monitoring frequency", "Prepare for possible escalation of care"]
        if latest_dict.get("spo2", 100) < 92:
            actions.append("Increase oxygen therapy")
        if latest_dict.get("blood_pressure_sys", 120) < 90:
            actions.append("Consider vasopressor therapy")
    elif risk["score"] >= 40:
        actions = ["Increase monitoring to every 15 minutes", "Review medication orders", "Notify charge nurse"]
    else:
        actions = ["Continue routine monitoring", "No immediate action required"]

    sepsis_prob = min(risk["score"] + 8, 100)
    deterioration_prob = min(risk["score"] + 3, 100)

    return {
        "patient_id": patient_id,
        "name": patient.name,
        "diagnosis": patient.diagnosis,
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
            "last_vitals_at": latest_dict["timestamp"],
        },
    }


# ── GET /icu/ai/predict/los/{patient_id} ──────────────────────────────────────
@router.get("/predict/los/{patient_id}", summary="Predict patient Length of Stay")
async def predict_los(patient_id: str, current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Predicts how many more days the patient is likely to remain in the ICU.
    """
    query = await db.execute(select(Patient).where(Patient.patient_id == patient_id))
    patient = query.scalar_one_or_none()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    v_query = await db.execute(select(Vital).where(Vital.patient_id == patient_id).order_by(Vital.timestamp.desc()).limit(1))
    latest = v_query.scalar_one_or_none()

    latest_dict = {}
    if latest:
        latest_dict = {
            "heart_rate": latest.heart_rate,
            "spo2": latest.spo2,
        }

    los = _predict_los(latest_dict, patient.age, patient.diagnosis)

    key_factors = []
    if patient.age and patient.age > 65: key_factors.append("Age > 65")
    if latest_dict.get("spo2", 100) < 92: key_factors.append("SpO2 < 92%")
    if patient.diagnosis and "sepsis" in patient.diagnosis.lower(): key_factors.append("Diagnosis: Sepsis")
    if patient.diagnosis and "respiratory" in patient.diagnosis.lower(): key_factors.append("Diagnosis: Respiratory Failure")

    return {
        "patient_id": patient_id,
        "name": patient.name,
        "diagnosis": patient.diagnosis,
        "predicted_los": los,
        "key_factors": key_factors if key_factors else ["Standard case — no major risk factors detected"],
    }


# ── GET /icu/ai/alerts ─────────────────────────────────────────────────────────
@router.get("/alerts", summary="Get all active AI-generated risk alerts")
async def get_ai_alerts(current_user=Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """
    Scans all ICU patients and returns those flagged by the AI risk model.
    """
    alerts = []
    
    query = await db.execute(select(Patient))
    patients = query.scalars().all()
    
    for patient in patients:
        v_query = await db.execute(select(Vital).where(Vital.patient_id == patient.patient_id).order_by(Vital.timestamp.desc()).limit(1))
        latest = v_query.scalar_one_or_none()
        
        if not latest:
            continue
            
        latest_dict = {
            "heart_rate": latest.heart_rate,
            "blood_pressure_sys": latest.blood_pressure_sys,
            "spo2": latest.spo2,
            "respiratory_rate": latest.respiratory_rate,
            "temperature": latest.temperature,
            "timestamp": latest.timestamp.isoformat() if latest.timestamp else None
        }
        
        risk = _compute_risk_score(latest_dict)
        if risk["score"] >= 40:
            factors = []
            if latest_dict.get("spo2", 100) < 94:
                factors.append(f"SpO2 low: {latest_dict['spo2']}%")
            if latest_dict.get("heart_rate", 80) > 100:
                factors.append(f"Tachycardia: {latest_dict['heart_rate']} bpm")
            if latest_dict.get("blood_pressure_sys", 120) < 100:
                factors.append(f"Low BP: {latest_dict['blood_pressure_sys']} mmHg")

            alerts.append({
                "patient_id": patient.patient_id,
                "name": patient.name,
                "bed_id": patient.bed_id,
                "risk_score": risk["score"],
                "category": risk["category"],
                "top_factor": factors[0] if factors else "Multiple parameters out of range",
                "generated_at": datetime.utcnow().isoformat(),
            })

    alerts.sort(key=lambda x: x["risk_score"], reverse=True)
    return {"total_alerts": len(alerts), "alerts": alerts}
