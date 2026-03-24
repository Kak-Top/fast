"""
Chatbot Router — General-Purpose AI with ICU Context Awareness
--------------------------------------------------------------
Uses qwen/qwen3-coder:free via OpenRouter API.
Can answer ANYTHING, but is also aware of the ICU state.
Falls back to keyword engine if API call fails.

Install:
    pip install httpx

Set env variable:
    OPENROUTER_API_KEY=your_key_here
"""

import os
import httpx
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from datetime import datetime
from dependencies import (
    fake_patients_db,
    fake_vitals_db,
    fake_resources_db,
    fake_siem_events_db,
    get_current_user,
)

router = APIRouter()

# In-memory session history per session_id
chat_sessions: dict = {}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# Model chain — tried in order on 429/404.
# openrouter/free is OpenRouter's own smart router that auto-picks
# any available free model, so it handles availability automatically.
FREE_MODELS = [
    "openrouter/free",              # auto-picks best available free model
    "qwen/qwen3-coder:free",        # fallback if router itself is down
    "google/gemma-3-12b-it:free",   # verified free March 2026
    "meta-llama/llama-3.3-70b-instruct:free",  # verified free March 2026
    "deepseek/deepseek-chat-v3-0324:free",     # verified free March 2026
]
FREE_MODEL = FREE_MODELS[0]  # updated to whichever model actually responds

# Max turns to keep in memory per session
MAX_HISTORY = 10


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class ChatQuery(BaseModel):
    question:   str
    session_id: str = "default"


# ── ICU Context builder ───────────────────────────────────────────────────────
def _build_icu_context() -> str:
    """
    Builds a real-time snapshot of the ICU state to inject as background
    context into the system prompt. The model can use this if asked,
    but is not restricted to it.
    """
    patients  = list(fake_patients_db.values())
    resources = list(fake_resources_db.values())
    siem      = fake_siem_events_db[-5:] if fake_siem_events_db else []

    critical = [p for p in patients if p.get("status") == "critical"]
    stable   = [p for p in patients if p.get("status") == "stable"]

    patient_vitals = []
    for p in patients:
        history = fake_vitals_db.get(p["patient_id"], [])
        if history:
            v = history[-1]
            patient_vitals.append(
                f"  - {p['name']} ({p['bed_id']}): "
                f"HR={v.get('heart_rate')} bpm, "
                f"SpO2={v.get('spo2')}%, "
                f"BP={v.get('blood_pressure_sys')}/{v.get('blood_pressure_dia')} mmHg, "
                f"Temp={v.get('temperature')}°C, "
                f"RR={v.get('respiratory_rate')} breaths/min — "
                f"Status: {p['status'].upper()}"
            )

    beds  = [r for r in resources if r["type"] == "bed"]
    vents = [r for r in resources if r["type"] == "ventilator"]
    mons  = [r for r in resources if r["type"] == "monitor"]

    beds_avail  = sum(1 for r in beds  if r["status"] == "available")
    vents_avail = sum(1 for r in vents if r["status"] == "available")
    mons_avail  = sum(1 for r in mons  if r["status"] == "available")

    siem_summary = "\n".join([
        f"  - [{e.get('severity')}] {e.get('event_type')}: {e.get('description')} from {e.get('source_ip')}"
        for e in siem
    ]) or "  - No recent security events"

    return f"""
=== LIVE ICU STATE (use when relevant) ===
PATIENTS ({len(patients)} total | Critical: {len(critical)} | Stable: {len(stable)})
  Critical: {', '.join(p['name'] for p in critical) or 'None'}
  Stable:   {', '.join(p['name'] for p in stable) or 'None'}

LATEST VITALS:
{chr(10).join(patient_vitals) or '  - No vitals recorded yet'}

RESOURCES:
  Beds:        {beds_avail}/{len(beds)} available
  Ventilators: {vents_avail}/{len(vents)} available
  Monitors:    {mons_avail}/{len(mons)} available

RECENT SECURITY EVENTS (last 5):
{siem_summary}
==========================================
"""


# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt() -> str:
    icu_context = _build_icu_context()
    return f"""You are a highly capable AI assistant embedded inside an ICU Digital Twin hospital system.

You can answer ANY question the user asks — whether it's about medicine, technology, programming, general knowledge, math, language, writing, or anything else. You are not restricted to ICU topics.

However, you also have real-time access to the current ICU state (patients, vitals, resources, security events). When a question relates to the ICU, use the live data below to give accurate, specific answers. Never invent patient names or fabricate values.

When answering:
- Be conversational and natural, like a knowledgeable colleague.
- For ICU/medical questions: be precise, clinical, and use the live data.
- For general questions: be helpful, clear, and thorough.
- If you're unsure about something, say so honestly.
- Keep responses concise unless depth is needed.

{icu_context}"""


# ── OpenRouter LLM call ───────────────────────────────────────────────────────
async def _ask_llm(question: str, session_id: str) -> str:
    """
    Sends question + ICU context + conversation history to OpenRouter.
    Tries each model in FREE_MODELS in order if one returns 429 (rate-limited).
    """
    global FREE_MODEL

    if not OPENROUTER_API_KEY:
        return _fallback_engine(question)

    # Rebuild system prompt each call so ICU data is always fresh
    system_prompt = _build_system_prompt()

    # Build message list: system + history + current question
    history = chat_sessions.get(session_id, [])
    messages = [{"role": "system", "content": system_prompt}]

    for entry in history[-MAX_HISTORY:]:
        messages.append({"role": "user",      "content": entry["question"]})
        messages.append({"role": "assistant",  "content": entry["answer"]})

    messages.append({"role": "user", "content": question})

    async with httpx.AsyncClient(timeout=45) as client:
        for model in FREE_MODELS:
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "http://capstone.dpdns.org",
                        "X-Title":       "ICU Digital Twin",
                    },
                    json={
                        "model":       model,
                        "messages":    messages,
                        "max_tokens":  600,
                        "temperature": 0.6,
                    },
                )
                data = response.json()

                if "choices" in data:
                    FREE_MODEL = model  # track which model actually responded
                    print(f"✓ Model used: {model}")
                    return data["choices"][0]["message"]["content"].strip()

                error = data.get("error", {})
                code  = error.get("code") if isinstance(error, dict) else None

                if code == 429:
                    print(f"429 rate-limit on {model}, trying next...")
                    continue  # try the next model
                else:
                    print(f"OpenRouter error on {model}: {error}")
                    continue  # non-429 error, still try next

            except Exception as e:
                print(f"Request failed for {model}: {e}")
                continue

    # All models exhausted
    print("All free models rate-limited or failed.")
    return _fallback_engine(question)


# ── Fallback keyword engine ───────────────────────────────────────────────────
def _fallback_engine(question: str) -> str:
    """Used only when the API key is missing or the call fails."""
    q = question.lower()

    if any(w in q for w in ["bed", "beds", "capacity"]):
        beds      = [r for r in fake_resources_db.values() if r["type"] == "bed"]
        available = sum(1 for b in beds if b["status"] == "available")
        occupied  = len(beds) - available
        return f"There are {len(beds)} ICU beds — {available} available, {occupied} occupied."

    if "ventilator" in q:
        vents     = [r for r in fake_resources_db.values() if r["type"] == "ventilator"]
        available = sum(1 for v in vents if v["status"] == "available")
        return f"There are {len(vents)} ventilators — {available} available."

    if any(w in q for w in ["critical", "deteriorat", "at risk", "danger"]):
        critical = [p for p in fake_patients_db.values() if p["status"] == "critical"]
        if critical:
            names = ", ".join(p["name"] for p in critical)
            return f"{len(critical)} critical patient(s): {names}."
        return "No patients are currently flagged as critical."

    if "sepsis" in q or "risk" in q:
        high_risk = [
            patient["name"]
            for pid, patient in fake_patients_db.items()
            if fake_vitals_db.get(pid) and (
                fake_vitals_db[pid][-1].get("spo2", 100) < 92
                or fake_vitals_db[pid][-1].get("blood_pressure_sys", 120) < 90
            )
        ]
        if high_risk:
            return f"High sepsis risk: {', '.join(high_risk)}. Immediate review recommended."
        return "No patients currently flagged as high sepsis risk."

    if any(w in q for w in ["summary", "status", "overview"]):
        beds_avail  = sum(1 for r in fake_resources_db.values() if r["type"] == "bed"        and r["status"] == "available")
        vents_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "ventilator" and r["status"] == "available")
        return (
            f"ICU Summary — Patients: {len(fake_patients_db)}, "
            f"Beds available: {beds_avail}, Ventilators available: {vents_avail}."
        )

    # Generic fallback for non-ICU questions when API is down
    return (
        "⚠️ AI model unavailable (no API key or connection failed). "
        "I can still answer ICU questions about: bed availability, ventilators, "
        "critical patients, vitals, sepsis risk, or give an ICU summary. "
        "For all other questions, please ensure OPENROUTER_API_KEY is set."
    )


# ── POST /chatbot/query ───────────────────────────────────────────────────────
@router.post("/query", summary="Ask the AI assistant anything")
async def chatbot_query(body: ChatQuery, current_user=Depends(get_current_user)):
    """
    Send any question to the AI assistant.
    It can answer general knowledge, medical, technical, or ICU-specific questions.
    Powered by Qwen3-Coder via OpenRouter (free tier).
    Falls back to keyword engine if API is unavailable.

    **Sample Requests:**
    ```json
    { "question": "Which patients are at risk right now?", "session_id": "session_dr_ahmad" }
    { "question": "Explain what sepsis is", "session_id": "session_dr_ahmad" }
    { "question": "Write a Python function to sort a list", "session_id": "session_dr_ahmad" }
    { "question": "What's the capital of France?", "session_id": "session_dr_ahmad" }
    ```
    """
    answer = await _ask_llm(body.question, body.session_id)

    # Store in session memory
    if body.session_id not in chat_sessions:
        chat_sessions[body.session_id] = []

    entry = {
        "question":  body.question,
        "answer":    answer,
        "timestamp": datetime.utcnow().isoformat(),
        "asked_by":  current_user["username"],
    }
    chat_sessions[body.session_id].append(entry)

    # Trim history to avoid unbounded growth
    if len(chat_sessions[body.session_id]) > MAX_HISTORY * 2:
        chat_sessions[body.session_id] = chat_sessions[body.session_id][-MAX_HISTORY:]

    return {
        **entry,
        "session_id": body.session_id,
        "model": FREE_MODEL if OPENROUTER_API_KEY else "fallback-keyword-engine",
    }


# ── GET /chatbot/history ──────────────────────────────────────────────────────
@router.get("/history", summary="Get chat session history")
def get_history(session_id: str = "default", current_user=Depends(get_current_user)):
    """Returns the full conversation history for a given session."""
    history = chat_sessions.get(session_id, [])
    return {
        "session_id":     session_id,
        "total_messages": len(history),
        "history":        history,
    }


# ── DELETE /chatbot/history ───────────────────────────────────────────────────
@router.delete("/history", summary="Clear chatbot session history")
def clear_history(session_id: str = "default", current_user=Depends(get_current_user)):
    """Clears the conversation history for a session (fresh start)."""
    chat_sessions.pop(session_id, None)
    return {
        "message":    "Session history cleared.",
        "session_id": session_id,
        "cleared_by": current_user["username"],
    }
