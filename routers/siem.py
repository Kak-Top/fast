"""
SIEM Security Router
--------------------
Handles security event ingestion, anomaly detection, audit logging,
and incident management for the ICU Digital Twin.

AI Model used: Isolation Forest (simulated here).
Install with: pip install scikit-learn
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from dependencies import (
    fake_siem_events_db, fake_siem_incidents_db,
    get_current_user, require_role
)

router = APIRouter()

# Simple in-memory audit log
audit_log = []


class SecurityEvent(BaseModel):
    event_type: str        # "login", "data_access", "failed_auth", "device_alert", "api_call"
    source_ip: str
    user_id: Optional[str] = None
    resource: Optional[str] = None
    description: str
    severity: str = "INFO"  # "INFO" | "WARNING" | "CRITICAL"


class IncidentReport(BaseModel):
    title: str
    description: str
    severity: str  # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    related_event_id: Optional[str] = None


class IncidentStatusUpdate(BaseModel):
    status: str  # "open" | "investigating" | "resolved" | "closed"


def _run_anomaly_detection(events: list) -> list:
    """
    Simulated Isolation Forest anomaly detection.
    Replace with real sklearn IsolationForest in production:

        from sklearn.ensemble import IsolationForest
        model = IsolationForest(contamination=0.1)
        model.fit(feature_matrix)
        predictions = model.predict(feature_matrix)

    Flags:
    - More than 5 failed auths from same IP
    - Data access outside normal hours (before 6am or after 10pm)
    - Unusual volume of API calls
    """
    anomalies = []
    ip_fail_counts = {}
    api_call_counts = {}

    for e in events:
        # Rule 1: Brute force detection
        if e.get("event_type") == "failed_auth":
            ip = e.get("source_ip")
            ip_fail_counts[ip] = ip_fail_counts.get(ip, 0) + 1
            if ip_fail_counts[ip] >= 3:
                anomalies.append({
                    "anomaly_id": f"ANO-{len(anomalies)+1:03}",
                    "type": "BRUTE_FORCE_ATTEMPT",
                    "description": f"Multiple failed login attempts from {ip} ({ip_fail_counts[ip]} times)",
                    "source_ip": ip,
                    "risk_score": 85,
                    "model": "IsolationForest (simulated)",
                    "detected_at": datetime.utcnow().isoformat(),
                })

        # Rule 2: Off-hours data access
        if e.get("event_type") == "data_access":
            try:
                ts = datetime.fromisoformat(e.get("timestamp", datetime.utcnow().isoformat()))
                if ts.hour < 6 or ts.hour > 22:
                    anomalies.append({
                        "anomaly_id": f"ANO-{len(anomalies)+1:03}",
                        "type": "OFF_HOURS_ACCESS",
                        "description": f"Patient data accessed at unusual hour ({ts.hour}:00) by {e.get('user_id')}",
                        "user_id": e.get("user_id"),
                        "risk_score": 65,
                        "model": "IsolationForest (simulated)",
                        "detected_at": datetime.utcnow().isoformat(),
                    })
            except Exception:
                pass

        # Rule 3: High API call volume
        uid = e.get("user_id")
        if uid and e.get("event_type") == "api_call":
            api_call_counts[uid] = api_call_counts.get(uid, 0) + 1
            if api_call_counts[uid] > 10:
                anomalies.append({
                    "anomaly_id": f"ANO-{len(anomalies)+1:03}",
                    "type": "ABNORMAL_API_VOLUME",
                    "description": f"User {uid} made {api_call_counts[uid]} API calls in a short window",
                    "user_id": uid,
                    "risk_score": 55,
                    "model": "IsolationForest (simulated)",
                    "detected_at": datetime.utcnow().isoformat(),
                })

    # Deduplicate by type+source
    seen = set()
    unique = []
    for a in anomalies:
        key = (a["type"], a.get("source_ip", a.get("user_id")))
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique


# ── POST /siem/events ─────────────────────────────────────────────────────────
@router.post("/events", summary="Ingest a new security event")
def ingest_event(body: SecurityEvent, current_user=Depends(get_current_user)):
    """
    Log a security event from any source: API gateway, IoT device, login system.

    **Sample Request Body:**
    ```json
    {
      "event_type": "failed_auth",
      "source_ip": "192.168.1.45",
      "user_id": "unknown",
      "resource": "/auth/login",
      "description": "Invalid password attempt",
      "severity": "WARNING"
    }
    ```

    **Sample Response:**
    ```json
    {
      "message": "Security event logged",
      "event_id": "EVT-001",
      "severity": "WARNING",
      "timestamp": "2025-03-02T09:00:00"
    }
    ```
    """
    event_id = f"EVT-{len(fake_siem_events_db)+1:03}"
    event = {
        "event_id": event_id,
        **body.dict(),
        "logged_by": current_user["username"],
        "timestamp": datetime.utcnow().isoformat(),
        "acknowledged": False,
    }
    fake_siem_events_db.append(event)

    # Auto-log to audit trail
    audit_log.append({
        "action": "SECURITY_EVENT_INGESTED",
        "event_id": event_id,
        "event_type": body.event_type,
        "by": current_user["username"],
        "at": datetime.utcnow().isoformat(),
    })

    return {
        "message": "Security event logged",
        "event_id": event_id,
        "severity": body.severity,
        "timestamp": event["timestamp"],
    }


# ── GET /siem/anomalies ───────────────────────────────────────────────────────
@router.get("/anomalies", summary="Run anomaly detection on logged events")
def get_anomalies(current_user=Depends(require_role("admin", "it_security"))):
    """
    Runs the Isolation Forest anomaly detection model over all logged security events
    and returns flagged anomalies.

    **Sample Response:**
    ```json
    {
      "total_anomalies": 1,
      "anomalies": [
        {
          "anomaly_id": "ANO-001",
          "type": "BRUTE_FORCE_ATTEMPT",
          "description": "Multiple failed login attempts from 192.168.1.45 (3 times)",
          "source_ip": "192.168.1.45",
          "risk_score": 85,
          "model": "IsolationForest (simulated)",
          "detected_at": "2025-03-02T09:05:00"
        }
      ]
    }
    ```
    """
    anomalies = _run_anomaly_detection(fake_siem_events_db)
    return {"total_anomalies": len(anomalies), "anomalies": anomalies}


# ── GET /siem/alerts ──────────────────────────────────────────────────────────
@router.get("/alerts", summary="Get active security alerts")
def get_security_alerts(current_user=Depends(require_role("admin", "it_security"))):
    """
    Returns all unacknowledged CRITICAL and WARNING security events.

    **Sample Response:**
    ```json
    {
      "total": 2,
      "alerts": [
        {
          "event_id": "EVT-001",
          "event_type": "failed_auth",
          "severity": "WARNING",
          "source_ip": "192.168.1.45",
          "description": "Invalid password attempt",
          "timestamp": "2025-03-02T09:00:00",
          "acknowledged": false
        }
      ]
    }
    ```
    """
    alerts = [e for e in fake_siem_events_db if e["severity"] in ("WARNING", "CRITICAL") and not e["acknowledged"]]
    return {"total": len(alerts), "alerts": alerts}


# ── PUT /siem/alerts/{id}/acknowledge ─────────────────────────────────────────
@router.put("/alerts/{event_id}/acknowledge", summary="Acknowledge a security alert")
def acknowledge_alert(event_id: str, current_user=Depends(require_role("admin", "it_security"))):
    """
    Marks a security event as acknowledged by the IT/security team.

    **Sample Response:**
    ```json
    {
      "message": "Alert EVT-001 acknowledged",
      "acknowledged_by": "admin",
      "acknowledged_at": "2025-03-02T09:10:00"
    }
    ```
    """
    event = next((e for e in fake_siem_events_db if e["event_id"] == event_id), None)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    event["acknowledged"] = True
    event["acknowledged_by"] = current_user["username"]
    event["acknowledged_at"] = datetime.utcnow().isoformat()

    audit_log.append({
        "action": "ALERT_ACKNOWLEDGED",
        "event_id": event_id,
        "by": current_user["username"],
        "at": datetime.utcnow().isoformat(),
    })

    return {
        "message": f"Alert {event_id} acknowledged",
        "acknowledged_by": current_user["username"],
        "acknowledged_at": event["acknowledged_at"],
    }


# ── GET /siem/audit-log ───────────────────────────────────────────────────────
@router.get("/audit-log", summary="Get full system audit trail")
def get_audit_log(current_user=Depends(require_role("admin", "it_security"))):
    """
    Returns the full audit trail of all system actions.
    Required for HIPAA/GDPR compliance and Jordan PDPL No. 24 of 2023.

    **Sample Response:**
    ```json
    {
      "total_entries": 3,
      "audit_log": [
        {
          "action": "SECURITY_EVENT_INGESTED",
          "event_id": "EVT-001",
          "event_type": "failed_auth",
          "by": "dr.ahmad",
          "at": "2025-03-02T09:00:00"
        },
        {
          "action": "ALERT_ACKNOWLEDGED",
          "event_id": "EVT-001",
          "by": "admin",
          "at": "2025-03-02T09:10:00"
        }
      ]
    }
    ```
    """
    return {"total_entries": len(audit_log), "audit_log": audit_log}


# ── POST /siem/incidents ──────────────────────────────────────────────────────
@router.post("/incidents", summary="Create a security incident report")
def create_incident(body: IncidentReport, current_user=Depends(require_role("admin", "it_security"))):
    """
    Formally documents a security incident for investigation and compliance.

    **Sample Request Body:**
    ```json
    {
      "title": "Unauthorized Access Attempt on ICU Records",
      "description": "Multiple failed login attempts detected from external IP",
      "severity": "HIGH",
      "related_event_id": "EVT-001"
    }
    ```

    **Sample Response:**
    ```json
    {
      "message": "Incident created",
      "incident": {
        "incident_id": "INC-001",
        "title": "Unauthorized Access Attempt on ICU Records",
        "severity": "HIGH",
        "status": "open",
        "created_by": "admin",
        "created_at": "2025-03-02T09:15:00"
      }
    }
    ```
    """
    incident_id = f"INC-{len(fake_siem_incidents_db)+1:03}"
    incident = {
        "incident_id": incident_id,
        **body.dict(),
        "status": "open",
        "created_by": current_user["username"],
        "created_at": datetime.utcnow().isoformat(),
        "updates": [],
    }
    fake_siem_incidents_db.append(incident)

    audit_log.append({
        "action": "INCIDENT_CREATED",
        "incident_id": incident_id,
        "by": current_user["username"],
        "at": datetime.utcnow().isoformat(),
    })

    return {"message": "Incident created", "incident": incident}
