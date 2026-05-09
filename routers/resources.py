from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_, desc
import uuid
import pandas as pd
import numpy as np
import os
import warnings
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dependencies import get_current_user, require_role
from database import get_db
from models import Resource, Patient, Vital

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────────────────

class ResourceUpdate(BaseModel):
    """Model for updating resource status"""
    status: str  # "available" | "occupied" | "in_use" | "maintenance"
    patient_id: Optional[str] = None


class VitalsReading(BaseModel):
    """Model for vitals readings"""
    heart_rate: float
    blood_pressure_sys: float
    blood_pressure_dia: float
    spo2: float
    respiratory_rate: float
    temperature: float


class ResourceResponse(BaseModel):
    """Response model for resources"""
    resource_id: str
    type: str
    status: str
    patient_id: Optional[str] = None


class PatientResponse(BaseModel):
    """Response model for patients"""
    patient_id: str
    name: str
    age: int
    gender: str
    diagnosis: str
    bed_id: str
    status: str
    admitted_at: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL RANGES FOR VITALS
# ─────────────────────────────────────────────────────────────────────────────

NORMAL_RANGES = {
    "heart_rate": (60, 100),
    "blood_pressure_sys": (90, 140),
    "blood_pressure_dia": (60, 90),
    "spo2": (95, 100),
    "respiratory_rate": (12, 20),
    "temperature": (36.0, 37.5),
}


def is_critical(vitals: dict) -> bool:
    """Check if vitals are critical"""
    for key, (low, high) in NORMAL_RANGES.items():
        if key in vitals and vitals[key] is not None and not (low <= vitals[key] <= high):
            return True
    return False


def flag_abnormal_params(vitals: dict) -> list:
    """Flag abnormal vitals parameters"""
    flags = []
    for key, (low, high) in NORMAL_RANGES.items():
        if key in vitals and vitals[key] is not None and not (low <= vitals[key] <= high):
            flags.append({
                "parameter": key,
                "value": vitals[key],
                "normal_range": f"{low}–{high}",
                "severity": "CRITICAL" if abs(vitals[key] - (low + high) / 2) > (high - low) else "WARNING"
            })
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER SETUP
# ─────────────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/icu", tags=["ICU Resources & Management"])


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/resources", summary="Get ICU resource summary (beds, ventilators, monitors)")
async def get_resources(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns a full overview of all ICU resources grouped by type,
    with availability counts.
    
    ### Response:
    - **summary**: Resource counts grouped by type and status
    - **resources**: Full list of all resources with details
    """
    # Query actual database
    query = await db.execute(select(Resource))
    resources = query.scalars().all()

    # Build summary by type
    summary = {}
    resources_list = []
    
    for r in resources:
        t = r.type
        if t not in summary:
            summary[t] = {}
        s = r.status
        summary[t][s] = summary[t].get(s, 0) + 1
        summary[t]["total"] = summary[t].get("total", 0) + 1
        
        resources_list.append({
            "resource_id": r.resource_id,
            "type": r.type,
            "status": r.status,
            "patient_id": r.patient_id
        })

    return {
        "summary": summary,
        "resources": resources_list,
        "timestamp": datetime.utcnow().isoformat()
    }


@router.put("/resources/{resource_id}", summary="Update a resource status")
async def update_resource(
    resource_id: str,
    body: ResourceUpdate,
    current_user=Depends(require_role("clinician", "admin", "manager")),
    db: AsyncSession = Depends(get_db)
):
    """
    Update the status of any ICU resource (bed, ventilator, monitor).
    
    ### Path Parameters:
    - **resource_id**: The ID of the resource to update
    
    ### Request Body:
    - **status**: New status ("available" | "occupied" | "in_use" | "maintenance")
    - **patient_id**: Associated patient ID (optional)
    
    ### Response:
    - Confirmation with updated resource details
    """
    query = await db.execute(select(Resource).where(Resource.resource_id == resource_id))
    resource = query.scalar_one_or_none()
    
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")

    valid_statuses = {"available", "occupied", "in_use", "maintenance"}
    if body.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status. Choose from: {valid_statuses}")

    resource.status = body.status
    resource.patient_id = body.patient_id
    await db.commit()
    await db.refresh(resource)

    return {
        "message": "Resource updated",
        "resource": {
            "resource_id": resource.resource_id,
            "type": resource.type,
            "status": resource.status,
            "patient_id": resource.patient_id
        }
    }


@router.post("/resources", summary="Create a new ICU resource")
async def create_resource(
    resource_type: str,
    current_user=Depends(require_role("admin", "manager")),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new ICU resource (bed, ventilator, monitor).
    
    ### Query Parameters:
    - **resource_type**: Type of resource ("bed" | "ventilator" | "monitor")
    
    ### Response:
    - New resource details with generated resource_id
    """
    valid_types = {"bed", "ventilator", "monitor"}
    if resource_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid type. Choose from: {valid_types}")

    resource_id = f"{resource_type.upper()}-{uuid.uuid4().hex[:6].upper()}"

    new_resource = Resource(
        resource_id=resource_id,
        type=resource_type,
        status="available"
    )
    db.add(new_resource)
    await db.commit()
    await db.refresh(new_resource)

    return {
        "message": "Resource created",
        "resource": {
            "resource_id": new_resource.resource_id,
            "type": new_resource.type,
            "status": new_resource.status,
            "patient_id": new_resource.patient_id
        }
    }


@router.get("/resources/{resource_id}", summary="Get resource details")
async def get_resource_detail(
    resource_id: str,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed information about a specific resource.
    
    ### Path Parameters:
    - **resource_id**: The resource ID
    """
    query = await db.execute(select(Resource).where(Resource.resource_id == resource_id))
    resource = query.scalar_one_or_none()
    
    if not resource:
        raise HTTPException(status_code=404, detail="Resource not found")
    
    return {
        "resource_id": resource.resource_id,
        "type": resource.type,
        "status": resource.status,
        "patient_id": resource.patient_id,
        "created_at": datetime.utcnow().isoformat()
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION & WHAT-IF ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/simulation/whatif", summary="Run a what-if simulation scenario")
async def run_whatif(
    scenario: str,
    extra_beds: int = 0,
    extra_ventilators: int = 0,
    extra_staff: int = 0,
    surge_percent: int = 0,
    current_user=Depends(require_role("admin", "manager", "clinician")),
    db: AsyncSession = Depends(get_db)
):
    """
    Simulates a hypothetical scenario and returns projected capacity impact.
    
    ### Query Parameters:
    - **scenario**: Scenario name (e.g., "flu_surge", "equipment_failure")
    - **extra_beds**: Number of additional beds to model
    - **extra_ventilators**: Number of additional ventilators
    - **extra_staff**: Number of additional staff
    - **surge_percent**: Percentage increase in patient volume (0-100)
    
    ### Response:
    - Current and projected capacity metrics
    - Risk assessment and recommendations
    """
    # Query actual resources from database
    resources_query = await db.execute(select(Resource))
    resources = resources_query.scalars().all()
    
    beds_available = sum(1 for r in resources if r.type == "bed" and r.status == "available")
    vents_available = sum(1 for r in resources if r.type == "ventilator" and r.status == "available")

    patients_query = await db.execute(select(Patient))
    patients = patients_query.scalars().all()
    patient_count = len(patients)
    
    # Calculations
    projected_beds = beds_available + extra_beds
    projected_vents = vents_available + extra_ventilators
    expected_surge = int(patient_count * (1 + surge_percent / 100))
    shortage = expected_surge - (projected_beds + patient_count - beds_available)

    risk_level = "LOW"
    if shortage > 3:
        risk_level = "CRITICAL"
    elif shortage > 0:
        risk_level = "HIGH"
    elif projected_vents < 2:
        risk_level = "MEDIUM"

    return {
        "scenario": scenario,
        "inputs": {
            "surge_percent": surge_percent,
            "extra_beds": extra_beds,
            "extra_ventilators": extra_ventilators,
            "extra_staff": extra_staff,
        },
        "current_capacity": {
            "beds_available": beds_available,
            "ventilators_available": vents_available,
            "patients_admitted": patient_count,
        },
        "projected_capacity": {
            "beds_available": projected_beds,
            "ventilators_available": projected_vents,
        },
        "expected_patient_surge": expected_surge,
        "projected_shortage": max(shortage, 0),
        "recommendation": (
            f"With a {surge_percent}% surge and {extra_beds} extra beds, "
            f"projected shortage is {max(shortage, 0)} beds. "
            + ("Ventilator supply is tight." if projected_vents < 2 else "Ventilators are sufficient.")
        ),
        "risk_level": risk_level,
    }


@router.post("/simulation/capacity-planning", summary="Advanced capacity planning")
async def capacity_planning(
    days_ahead: int = 7,
    current_user=Depends(require_role("admin", "manager")),
    db: AsyncSession = Depends(get_db)
):
    """
    Run predictive capacity planning for the next N days.
    
    ### Query Parameters:
    - **days_ahead**: Number of days to forecast (1-30)
    
    ### Response:
    - Daily forecast of bed and equipment usage
    - Critical periods and recommendations
    """
    if days_ahead < 1 or days_ahead > 30:
        raise HTTPException(status_code=400, detail="days_ahead must be between 1 and 30")

    forecast = []
    for day in range(1, days_ahead + 1):
        forecast.append({
            "date": (datetime.utcnow() + timedelta(days=day)).isoformat(),
            "predicted_admissions": 2 + day % 3,
            "predicted_discharges": 1 + day % 2,
            "bed_utilization_percent": 60 + (day * 5) % 30,
            "ventilator_utilization_percent": 45 + (day * 3) % 25,
            "risk_level": "HIGH" if (60 + (day * 5) % 30) > 85 else "MEDIUM" if (60 + (day * 5) % 30) > 70 else "LOW",
        })

    return {
        "planning_horizon": days_ahead,
        "forecast": forecast,
        "summary": {
            "peak_day": max(forecast, key=lambda x: x["bed_utilization_percent"])["date"],
            "avg_bed_utilization": np.mean([f["bed_utilization_percent"] for f in forecast]),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD & ANALYTICS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/dashboard/summary", summary="Get ICU dashboard summary")
async def get_dashboard_summary(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Returns comprehensive ICU dashboard with current status and key metrics.
    """
    # Query from database
    resources_query = await db.execute(select(Resource))
    resources = resources_query.scalars().all()
    
    patients_query = await db.execute(select(Patient))
    patients = patients_query.scalars().all()
    
    critical_patients_query = await db.execute(
        select(Patient).where(Patient.status == "critical")
    )
    critical_patients = critical_patients_query.scalars().all()
    
    # Count resources by type and status
    beds = {"total": 0, "available": 0, "occupied": 0, "maintenance": 0}
    ventilators = {"total": 0, "available": 0, "in_use": 0, "maintenance": 0}
    monitors = {"total": 0, "available": 0, "in_use": 0}
    
    for r in resources:
        if r.type == "bed":
            beds["total"] += 1
            beds[r.status] = beds.get(r.status, 0) + 1
        elif r.type == "ventilator":
            ventilators["total"] += 1
            ventilators[r.status] = ventilators.get(r.status, 0) + 1
        elif r.type == "monitor":
            monitors["total"] += 1
            monitors[r.status] = monitors.get(r.status, 0) + 1
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "total_patients": len(patients),
        "critical_patients": len(critical_patients),
        "beds": beds,
        "ventilators": ventilators,
        "monitors": monitors,
        "staff_on_duty": {
            "doctors": 4,
            "nurses": 8,
            "respiratory_therapists": 3,
        }
    }


@router.get("/analytics/resource-utilization", summary="Get resource utilization analytics")
async def get_resource_utilization(
    hours: int = 24,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get resource utilization trends over the specified period.
    
    ### Query Parameters:
    - **hours**: Number of hours of history to retrieve (1-720)
    """
    if hours < 1 or hours > 720:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 720")

    timeline = []
    for i in range(hours):
        timeline.append({
            "timestamp": (datetime.utcnow() - timedelta(hours=hours-i)).isoformat(),
            "bed_utilization": 60 + (i * 2) % 30,
            "ventilator_utilization": 45 + (i % 25),
            "monitor_utilization": 70 + (i % 20),
        })

    return {
        "period_hours": hours,
        "timeline": timeline,
        "average_utilization": {
            "beds": np.mean([t["bed_utilization"] for t in timeline]),
            "ventilators": np.mean([t["ventilator_utilization"] for t in timeline]),
            "monitors": np.mean([t["monitor_utilization"] for t in timeline]),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY: ICU Dashboard Backend Class (for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class ICUDashboardBackend:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.feature_importance = None
        self.day_based_model = None
        self.day_based_scaler = None
        self.historical_patterns = None
        
    def load_or_generate_data(self):
        """Load existing data or generate synthetic data"""
        try:
            # Try to load existing data (local to separated dashboard)
            self.data = pd.read_csv('datasets/icu_managment_datasets/icu_resource_data.csv')
            # Load models
            try:
                model_data = joblib.load('models/resource_model.pkl')
                self.model = model_data['model']
                self.scaler = model_data['scaler']
            except:
                # If models don't exist, train them
                self.train_models()
            return True
        except FileNotFoundError:
            # Generate synthetic data if no existing data found
            self.generate_synthetic_data()
            self.train_models()
            # Save the generated data for future use (local to separated dashboard)
            os.makedirs('datasets/icu_managment_datasets', exist_ok=True)
            self.data.to_csv('datasets/icu_managment_datasets/icu_resource_data.csv', index=False)
            return True
    
    def train_models(self):
        """Train ML models for predictions"""
        try:
            # Prepare features for training
            features = ['patients', 'available_doctors', 'available_nurses', 
                       'available_respiratory_therapists', 'available_technicians',
                       'available_ventilators', 'available_monitors']
            
            X = self.data[features]
            y = self.data['overall_busyness']
            
            # Train RandomForest model
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            import joblib
            
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            
            # Save models
            os.makedirs('models', exist_ok=True)
            joblib.dump({'model': self.model, 'scaler': self.scaler}, 'models/resource_model.pkl')
            
            # Train day-based model
            self.train_day_based_model()
            
            return True
        except Exception as e:
            print(f"Error training models: {e}")
            return False
    
    def get_current_status(self):
        """Get current ICU status"""
        current_date = self.data['date'].max()
        return self.data[self.data['date'] == current_date].iloc[0], current_date
    
    def predict_scenario_impact(self, current_status, scenarios):
        """Predict impact of different scenarios"""
        current_resources = {
            'available_doctors': current_status['available_doctors'],
            'available_nurses': current_status['available_nurses'],
            'available_respiratory_therapists': current_status['available_respiratory_therapists'],
            'available_technicians': current_status['available_technicians'],
            'available_ventilators': current_status['available_ventilators'],
            'available_monitors': current_status['available_monitors'],
            'patients': current_status['patients'],
            'overall_busyness': current_status['overall_busyness']
        }
        
        predictions = {'Current': current_resources.copy()}
        
        for scenario_name, changes in scenarios.items():
            scenario = current_resources.copy()
            for resource, value in changes.items():
                # For custom scenario, use absolute values; for predefined scenarios, use relative changes
                if scenario_name == 'Custom Scenario':
                    scenario[resource] = max(1, value)
                else:
                    # Predefined scenarios: apply as relative changes
                    if resource in current_resources:
                        scenario[resource] = max(1, current_resources[resource] + value)
                    else:
                        scenario[resource] = max(1, value)
            predictions[scenario_name] = scenario
        
        # Predict for each scenario
        results = {}
        for scenario_name, resources in predictions.items():
            features = [
                resources['available_doctors'],
                resources['available_nurses'],
                resources['available_respiratory_therapists'],
                resources['available_technicians'],
                resources['available_ventilators'],
                resources['available_monitors'],
                resources['patients']
            ]
            
            features_scaled = self.scaler.transform([features])
            predicted_busyness = self.model.predict(features_scaled)[0]
            results[scenario_name] = predicted_busyness
        
        return results
    
    def generate_insights_and_recommendations(self, current_status, predictions):
        """Generate actionable insights and recommendations based on current and predicted scenarios"""
        
        baseline = predictions['Current']
        insights = []
        recommendations = []
        actions = []
        
        # Analyze current status
        current_busyness = current_status['overall_busyness']
        
        # Current workload insights
        if current_busyness > 2.0:
            insights.append(f" Critical workload: Current busyness of {current_busyness:.2f} exceeds safe thresholds")
            recommendations.append("Implement emergency staffing protocols immediately")
            actions.append("Activate on-call staff and consider patient transfers to lower-acuity units")
        elif current_busyness > 1.5:
            insights.append(f" Elevated workload: Current busyness of {current_busyness:.2f} is above optimal levels")
            recommendations.append("Review staffing allocation and consider overtime options")
            actions.append("Monitor patient acuity closely and prepare contingency plans")
        else:
            insights.append(f" Optimal workload: Current busyness of {current_busyness:.2f} is within safe ranges")
            recommendations.append("Maintain current staffing levels")
            actions.append("Schedule routine maintenance and staff training sessions")
        
        # Staff-specific insights
        nurse_ratio = current_status['patients'] / current_status['available_nurses']
        if nurse_ratio > 3:
            insights.append(f" Critical nurse-to-patient ratio: {nurse_ratio:.1f}:1 exceeds recommended 3:1")
            recommendations.append("Immediately increase nursing staff or redistribute patient load")
            actions.append("Consider temporary nursing agency staff and reduce elective admissions")
        elif nurse_ratio > 2.5:
            insights.append(f" Elevated nurse workload: {nurse_ratio:.1f}:1 approaches recommended limits")
            recommendations.append("Monitor nursing workload closely and prepare backup plans")
            actions.append("Assign additional support staff to nursing teams")
        
        # Equipment stress analysis
        ventilator_stress = current_status['patients'] / max(1, current_status['available_ventilators'])
        if ventilator_stress > 0.8:
            insights.append(f" High ventilator utilization: {ventilator_stress*100:.0f}% capacity in use")
            recommendations.append("Review ventilator allocation protocols and maintenance schedules")
            actions.append("Prepare backup ventilators and consider early liberation strategies")
        elif ventilator_stress > 0.6:
            insights.append(f" Moderate ventilator utilization: {ventilator_stress*100:.0f}% capacity")
            recommendations.append("Monitor ventilator availability and maintenance status")
            actions.append("Schedule preventive maintenance for backup equipment")
        
        # Scenario impact analysis
        worst_scenario = max(predictions.items(), key=lambda x: x[1] if x[0] != 'Current' else -float('inf'))
        best_scenario = min(predictions.items(), key=lambda x: x[1] if x[0] != 'Current' else float('inf'))
        
        if worst_scenario[0] != 'Current':
            worst_change = worst_scenario[1] - baseline
            worst_change_pct = (worst_change / baseline) * 100
            insights.append(f"  Worst-case scenario: {worst_scenario[0]} could increase busyness by {worst_change_pct:.1f}%")
            recommendations.append("Develop contingency plans for high-risk scenarios")
            actions.append("Create rapid response protocols for identified risk factors")
        
        if best_scenario[0] != 'Current':
            best_change = best_scenario[1] - baseline
            best_change_pct = (best_change / baseline) * 100
            insights.append(f" Best-case opportunity: {best_scenario[0]} could reduce busyness by {abs(best_change_pct):.1f}%")
            recommendations.append("Implement resource optimization strategies")
            actions.append("Document successful approaches and create standard operating procedures")
        
        # Staff availability analysis (calculate from available vs expected)
        total_staff = current_status['available_doctors'] + current_status['available_nurses'] + current_status['available_respiratory_therapists'] + current_status['available_technicians']
        expected_staff = 20  # Expected total staff based on patient load
        staff_sick = max(0, expected_staff - total_staff)
        
        if staff_sick > 3:
            insights.append(f" High staff absences: {staff_sick} staff members absent today")
            recommendations.append("Review sick leave patterns and implement backup staffing")
            actions.append("Cross-train staff to cover multiple roles during absences")
        
        # Equipment maintenance analysis (calculate from available vs expected)
        expected_ventilators = current_status['patients'] + 5  # Expected ventilators
        equipment_maintenance = max(0, expected_ventilators - current_status['available_ventilators'])
        
        if equipment_maintenance > 2:
            insights.append(f" Multiple equipment under maintenance: {equipment_maintenance} units offline")
            recommendations.append("Prioritize critical equipment maintenance and expedite repairs")
            actions.append("Activate backup equipment and adjust resource allocation")
        
        # Seasonal adjustments
        current_date = datetime.strptime(current_status['date'], '%Y-%m-%d')
        current_month = current_date.month
        if current_month in [12, 1, 2]:  # Winter months
            insights.append(" Winter season: Typically higher patient volumes and complex cases")
            recommendations.append("Enhance staffing and prepare for seasonal surges")
            actions.append("Implement early discharge protocols and optimize bed management")
        
        # Resource optimization opportunities
        doctor_workload = current_status['patients'] / max(1, current_status['available_doctors'])
        if doctor_workload < 2.0 and doctor_workload > 1.0:
            insights.append(" Opportunity for doctor redistribution: Current workload allows for coverage expansion")
            recommendations.append("Consider reallocating doctor resources to other high-needs areas")
            actions.append("Assign doctors to consultation services or quality improvement projects")
        
        return insights, recommendations, actions
    
    def train_day_based_model(self):
        """Train a model to predict expected resources based on day of week"""
        try:
            # Create realistic day-based patterns for ICU resource needs
            # Based on typical hospital patterns: weekdays vs weekends
            
            day_patterns = pd.DataFrame({
                'day_of_week': range(7),  # 0=Monday, 6=Sunday
                # Patient count varies by day (higher mid-week, lower weekends)
                'patient_count': [18, 20, 22, 21, 19, 12, 10],
                # High-risk cases higher during weekdays
                'high_risk_count': [8, 9, 10, 9, 7, 4, 3],
                # Average acuity level
                'avg_acuity': [2.1, 2.2, 2.4, 2.3, 2.0, 1.6, 1.5]
            })
            
            # Calculate expected resources based on patient load and acuity
            day_patterns['expected_doctors'] = np.where(day_patterns['patient_count'] > 18, 7, 
                                                       np.where(day_patterns['patient_count'] > 14, 6, 5))
            day_patterns['expected_nurses'] = (day_patterns['patient_count'] * 1.2).astype(int) + 5
            day_patterns['expected_respiratory_therapists'] = np.where(day_patterns['high_risk_count'] > 7, 4,
                                                                      np.where(day_patterns['high_risk_count'] > 5, 3, 2))
            day_patterns['expected_technicians'] = (day_patterns['patient_count'] * 0.4).astype(int) + 3
            day_patterns['expected_ventilators'] = day_patterns['high_risk_count'] + 5
            day_patterns['expected_monitors'] = day_patterns['patient_count'] + 5
            
            # Calculate expected busyness based on patient-to-staff ratios and acuity
            day_patterns['expected_busyness'] = (
                day_patterns['patient_count'] / 15 +  # Base patient load
                day_patterns['high_risk_count'] / 10 +  # High-risk complexity
                day_patterns['avg_acuity'] / 2 +       # Overall acuity
                np.random.normal(0, 0.05, len(day_patterns))  # Small random variation
            ).round(2)
            
            # Add weekend adjustment factors
            weekend_days = [5, 6]  # Saturday, Sunday
            day_patterns.loc[day_patterns['day_of_week'].isin(weekend_days), 'expected_busyness'] *= 1.1
            
            self.historical_patterns = day_patterns
            
            # Train simple regression model for day-based predictions
            X = self.historical_patterns[['day_of_week']].values
            y_resources = self.historical_patterns[['expected_doctors', 'expected_nurses', 
                                                    'expected_respiratory_therapists', 'expected_technicians',
                                                    'expected_ventilators', 'expected_monitors']].values
            
            self.day_based_scaler = StandardScaler()
            X_scaled = self.day_based_scaler.fit_transform(X)
            
            self.day_based_model = {}
            resource_names = ['doctors', 'nurses', 'respiratory_therapists', 'technicians', 'ventilators', 'monitors']
            
            for i, resource in enumerate(resource_names):
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                model.fit(X_scaled, y_resources[:, i])
                self.day_based_model[resource] = model
                
        except Exception as e:
            raise Exception(f"Error training day-based model: {e}")
    
    def get_day_based_prediction(self, day_of_week):
        """Get predicted resources for a specific day of week"""
        if self.historical_patterns is None:
            return None
            
        # Get historical average for the specific day
        day_data = self.historical_patterns[self.historical_patterns['day_of_week'] == day_of_week]
        
        if len(day_data) > 0:
            # Get the first row of data
            first_row = day_data.iloc[0]
            
            # Handle patient count from different column names
            if 'patient_count' in first_row:
                patient_count = first_row['patient_count']
            elif 'expected_patients' in first_row:
                patient_count = first_row['expected_patients']
            else:
                patient_count = 15
            
            return {
                'predicted_doctors': int(round(first_row['expected_doctors'])),
                'predicted_nurses': int(round(first_row['expected_nurses'])),
                'predicted_respiratory_therapists': int(round(first_row['expected_respiratory_therapists'])),
                'predicted_technicians': int(round(first_row['expected_technicians'])),
                'predicted_ventilators': int(round(first_row['expected_ventilators'])),
                'predicted_monitors': int(round(first_row['expected_monitors'])),
                'predicted_busyness': float(first_row['expected_busyness']),
                'predicted_patients': int(round(patient_count))
            }
        else:
            # Return default values if no data for this day
            return {
                'predicted_doctors': 5,
                'predicted_nurses': 12,
                'predicted_respiratory_therapists': 3,
                'predicted_technicians': 6,
                'predicted_ventilators': 10,
                'predicted_monitors': 20,
                'predicted_busyness': 1.5,
                'predicted_patients': 15
            }
    
    def get_weekly_forecast(self):
        """Get predictions for the entire week"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_forecast = []
        
        for i, day_name in enumerate(days):
            prediction = self.get_day_based_prediction(i)
            if prediction:
                prediction['day_name'] = day_name
                prediction['day_of_week'] = i
                weekly_forecast.append(prediction)
        
        return weekly_forecast
    
    def get_monthly_trends(self):
        """Get monthly resource trends and predictions"""
        if self.data is None:
            return None
        
        # Group data by month and calculate averages
        monthly_data = self.data.copy()
        monthly_data['month'] = pd.to_datetime(monthly_data['date']).dt.month
        
        monthly_trends = monthly_data.groupby('month').agg({
            'patients': 'mean',
            'overall_busyness': 'mean',
            'available_doctors': 'mean',
            'available_nurses': 'mean',
            'available_ventilators': 'mean'
        }).round(2)
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        trends = []
        for month in range(1, 13):
            if month in monthly_trends.index:
                data = monthly_trends.loc[month]
                trends.append({
                    'month': month,
                    'month_name': month_names[month-1],
                    'avg_patients': float(data['patients']),
                    'avg_busyness': float(data['overall_busyness']),
                    'avg_doctors': float(data['available_doctors']),
                    'avg_nurses': float(data['available_nurses']),
                    'avg_ventilators': float(data['available_ventilators'])
                })
        
        return trends
    
    def get_peak_days_analysis(self):
        """Analyze peak and low demand days"""
        if self.historical_patterns is None:
            return None
        
        # Sort days by busyness
        sorted_days = self.historical_patterns.sort_values('expected_busyness', ascending=False)
        
        peak_days = []
        low_days = []
        
        for _, row in sorted_days.iterrows():
            day_info = {
                'day_of_week': int(row['day_of_week']),
                'day_name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][int(row['day_of_week'])],
                'predicted_patients': int(row['patient_count']),
                'predicted_busyness': float(row['expected_busyness']),
                'predicted_doctors': int(row['expected_doctors']),
                'predicted_nurses': int(row['expected_nurses'])
            }
            
            if len(peak_days) < 3:
                peak_days.append(day_info)
            elif len(low_days) < 3:
                low_days.append(day_info)
        
        return {
            'peak_days': peak_days,
            'low_days': low_days,
            'peak_day': peak_days[0] if peak_days else None,
            'lowest_day': low_days[-1] if low_days else None
        }
    
    def get_seasonal_predictions(self, current_date=None):
        """Get seasonal predictions based on time of year"""
        if current_date is None:
            current_date = datetime.now()
        
        month = current_date.month
        
        # Seasonal patterns (simplified)
        seasonal_factors = {
            'winter': [12, 1, 2],    # Higher respiratory issues
            'spring': [3, 4, 5],     # Moderate
            'summer': [6, 7, 8],     # Lower (vacations)
            'fall': [9, 10, 11]      # Increasing
        }
        
        season = None
        for season_name, months in seasonal_factors.items():
            if month in months:
                season = season_name
                break
        
        # Apply seasonal multipliers
        season_multipliers = {
            'winter': 1.2,  # 20% higher demand
            'spring': 1.0,  # Normal
            'summer': 0.9,  # 10% lower demand
            'fall': 1.1    # 10% higher demand
        }
        
        multiplier = season_multipliers.get(season, 1.0)
        
        # Get base prediction for current day of week
        day_of_week = current_date.weekday()
        base_prediction = self.get_day_based_prediction(day_of_week)
        
        if base_prediction:
            # Apply seasonal adjustment
            seasonal_prediction = base_prediction.copy()
            seasonal_prediction.update({
                'predicted_patients': int(base_prediction['predicted_patients'] * multiplier),
                'predicted_busyness': base_prediction['predicted_busyness'] * multiplier,
                'season': season,
                'seasonal_multiplier': multiplier
            })
            return seasonal_prediction
        
        return None
    
    def get_alerts(self, current_status):
        """Generate alerts based on current status"""
        alerts = []
        if current_status['overall_busyness'] > 2.0:
            alerts.append(" ICU is critically busy - consider calling in additional staff")
        elif current_status['overall_busyness'] > 1.5:
            alerts.append(" ICU is experiencing high workload")
        
        # Calculate ventilator stress
        ventilator_stress = current_status['patients'] / max(1, current_status['available_ventilators'])
        if ventilator_stress > 0.8:
            alerts.append(" Ventilator availability is critical")
        elif ventilator_stress > 0.6:
            alerts.append(" Ventilator usage is high")
        
        # Calculate nurse workload
        nurse_workload = current_status['patients'] / max(1, current_status['available_nurses'])
        if nurse_workload > 3:
            alerts.append(" Nurse workload exceeds safe levels")
        elif nurse_workload > 2:
            alerts.append(" Nurse workload is elevated")
        
        return alerts
    
    def get_efficiency_metrics(self, current_status):
        """Calculate efficiency metrics"""
        staff_efficiency = current_status['patients'] / (current_status['available_doctors'] + current_status['available_nurses'])
        equipment_efficiency = current_status['patients'] / (current_status['available_ventilators'] + current_status['available_monitors'])
        
        return {
            'staff_efficiency': staff_efficiency,
            'equipment_efficiency': equipment_efficiency,
            'capacity_utilization': current_status['overall_busyness'] * 100
        }
    
    def get_historical_comparison(self, current_status):
        """Get historical comparison data"""
        historical_avg = self.data['overall_busyness'].mean()
        historical_std = self.data['overall_busyness'].std()
        current_vs_avg = (current_status['overall_busyness'] - historical_avg) / historical_std
        
        return {
            'historical_avg': historical_avg,
            'current_vs_avg': current_vs_avg,
            'historical_std': historical_std
        }
