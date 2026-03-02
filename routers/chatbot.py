"""
Chatbot Router
--------------
Natural language interface for querying the ICU Digital Twin.

In production, replace the keyword-matching logic with:
    - LangChain + Claude/GPT API
    - Or a fine-tuned Llama/Mistral model

Install: pip install langchain openai anthropic
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime
from dependencies import fake_patients_db, fake_vitals_db, fake_resources_db, get_current_user

router = APIRouter()

# In-memory session history
chat_sessions: dict = {}


class ChatQuery(BaseModel):
    question: str
    session_id: str = "default"


def _simple_nlp_engine(question: str) -> str:
    """
    Keyword-based response engine (replace with LLM in production).
    """
    q = question.lower()

    # --- Bed / resource queries ---
    if any(word in q for word in ["bed", "beds", "available", "capacity"]):
        beds = [r for r in fake_resources_db.values() if r["type"] == "bed"]
        available = sum(1 for b in beds if b["status"] == "available")
        occupied = sum(1 for b in beds if b["status"] == "occupied")
        return (
            f"There are currently {len(beds)} ICU beds total. "
            f"{available} are available and {occupied} are occupied."
        )

    # --- Ventilator queries ---
    if "ventilator" in q:
        vents = [r for r in fake_resources_db.values() if r["type"] == "ventilator"]
        available = sum(1 for v in vents if v["status"] == "available")
        return f"There are {len(vents)} ventilators. {available} are currently available."

    # --- Critical patient queries ---
    if any(word in q for word in ["critical", "deteriorat", "at risk", "danger"]):
        critical = [p for p in fake_patients_db.values() if p["status"] == "critical"]
        if critical:
            names = ", ".join([p["name"] for p in critical])
            return f"There are {len(critical)} critical patient(s) right now: {names}."
        return "No patients are currently flagged as critical."

    # --- Patient count ---
    if any(word in q for word in ["how many patient", "total patient", "number of patient"]):
        return f"There are currently {len(fake_patients_db)} patients admitted to the ICU."

    # --- Specific patient vitals ---
    if "vitals" in q or "spo2" in q or "heart rate" in q or "blood pressure" in q:
        for pid, patient in fake_patients_db.items():
            if patient["name"].lower() in q or pid.lower() in q:
                history = fake_vitals_db.get(pid, [])
                if history:
                    v = history[-1]
                    return (
                        f"Latest vitals for {patient['name']}: "
                        f"HR={v.get('heart_rate')} bpm, "
                        f"BP={v.get('blood_pressure_sys')}/{v.get('blood_pressure_dia')} mmHg, "
                        f"SpO2={v.get('spo2')}%, "
                        f"RR={v.get('respiratory_rate')} bpm, "
                        f"Temp={v.get('temperature')}°C."
                    )
        return "Please specify a patient name or ID to retrieve vitals."

    # --- Sepsis risk ---
    if "sepsis" in q or "risk" in q:
        high_risk = []
        for pid, patient in fake_patients_db.items():
            history = fake_vitals_db.get(pid, [])
            if history:
                v = history[-1]
                if v.get("spo2", 100) < 92 or v.get("blood_pressure_sys", 120) < 90:
                    high_risk.append(patient["name"])
        if high_risk:
            return f"High sepsis risk detected for: {', '.join(high_risk)}. Immediate clinical review recommended."
        return "No patients are currently flagged as high sepsis risk."

    # --- Resource summary ---
    if "resource" in q or "summary" in q or "status" in q:
        beds_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "bed" and r["status"] == "available")
        vents_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "ventilator" and r["status"] == "available")
        return f"ICU Summary — Available beds: {beds_avail}, Available ventilators: {vents_avail}, Total patients: {len(fake_patients_db)}."

    # --- Default fallback ---
    return (
        "I'm sorry, I didn't understand that question. "
        "You can ask me about: bed availability, ventilators, critical patients, patient vitals, "
        "sepsis risk, or the ICU resource summary."
    )


# ── POST /chatbot/query ───────────────────────────────────────────────────────
@router.post("/query", summary="Ask the ICU Digital Twin a natural language question")
def chatbot_query(body: ChatQuery, current_user=Depends(get_current_user)):
    """
    Send a plain-language question to the ICU Digital Twin chatbot.

    **Example questions:**
    - "How many ICU beds are available?"
    - "Which patients are at risk right now?"
    - "What are Khalid's vitals?"
    - "Is there a ventilator available?"

    **Sample Request Body:**
    ```json
    {
      "question": "How many ICU beds are available?",
      "session_id": "session_dr_ahmad"
    }
    ```

    **Sample Response:**
    ```json
    {
      "session_id": "session_dr_ahmad",
      "question": "How many ICU beds are available?",
      "answer": "There are currently 3 ICU beds total. 1 is available and 2 are occupied.",
      "timestamp": "2025-03-02T09:30:00",
      "asked_by": "dr.ahmad"
    }
    ```
    """
    answer = _simple_nlp_engine(body.question)

    # Store in session history
    if body.session_id not in chat_sessions:
        chat_sessions[body.session_id] = []

    entry = {
        "question": body.question,
        "answer": answer,
        "timestamp": datetime.utcnow().isoformat(),
        "asked_by": current_user["username"],
    }
    chat_sessions[body.session_id].append(entry)

    return {"session_id": body.session_id, **entry}


# ── DELETE /chatbot/history ────────────────────────────────────────────────────
@router.delete("/history", summary="Clear chatbot session history")
def clear_history(session_id: str = "default", current_user=Depends(get_current_user)):
    """
    Clears the conversation history for a given session.

    **Sample Response:**
    ```json
    {
      "message": "Session history cleared",
      "session_id": "session_dr_ahmad",
      "cleared_by": "dr.ahmad"
    }
    ```
    """
    chat_sessions.pop(session_id, None)
    return {
        "message": "Session history cleared",
        "session_id": session_id,
        "cleared_by": current_user["username"],
    }
