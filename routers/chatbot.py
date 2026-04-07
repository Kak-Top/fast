"""
Chatbot Router — Fast + Clean Output
-------------------------------------
Fixes:
  1. Prioritizes fastest free models (GLM, GPT-OSS, MiniMax)
  2. Instructs model to respond in plain conversational text — no raw markdown tables
  3. Streams-friendly short responses
  4. Falls back cleanly

Install: pip install httpx
Env:     OPENROUTER_API_KEY=your_key_here
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

chat_sessions: dict = {}

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ── Model priority list — fastest first ───────────────────────────────────────
# Ranked by speed on free tier based on community benchmarks (March 2026)
FREE_MODELS = [
    "openrouter/free",
    "z-ai/glm-4.5-air:free",                        # fastest — GLM is very snappy
    "openai/gpt-oss-20b:free",                       # fast, GPT-style output quality
    "openai/gpt-oss-120b:free",                      # slower but highest quality
    "minimax/minimax-m2.5:free",                     # fast, good reasoning
    "qwen/qwen3-coder:free",                         # good but slower
    "meta-llama/llama-3.3-70b-instruct:free",        # reliable fallback
]

MAX_HISTORY = 7  # keep last 6 exchanges in memory


class ChatQuery(BaseModel):
    question:   str
    session_id: str = "default"


# ── ICU context — compact version ─────────────────────────────────────────────
def _build_icu_context() -> str:
    patients  = list(fake_patients_db.values())
    resources = list(fake_resources_db.values())

    critical = [p for p in patients if p.get("status") == "critical"]
    stable   = [p for p in patients if p.get("status") == "stable"]

    vitals_lines = []
    for p in patients:
        history = fake_vitals_db.get(p["patient_id"], [])
        if history:
            v = history[-1]
            vitals_lines.append(
                f"{p['name']} ({p['bed_id']}, {p['status'].upper()}): "
                f"HR {v.get('heart_rate')} bpm, SpO2 {v.get('spo2')}%, "
                f"BP {v.get('blood_pressure_sys')}/{v.get('blood_pressure_dia')} mmHg, "
                f"Temp {v.get('temperature')}°C, RR {v.get('respiratory_rate')}/min"
            )

    beds_avail  = sum(1 for r in resources if r["type"] == "bed"        and r["status"] == "available")
    vents_avail = sum(1 for r in resources if r["type"] == "ventilator" and r["status"] == "available")
    total_beds  = sum(1 for r in resources if r["type"] == "bed")
    total_vents = sum(1 for r in resources if r["type"] == "ventilator")

    recent_siem = fake_siem_events_db[-3:] if fake_siem_events_db else []
    siem_lines  = [f"[{e.get('severity')}] {e.get('event_type')}: {e.get('description')}" for e in recent_siem]

    lines = [
        f"ICU has {len(patients)} patients. Critical: {len(critical)} ({', '.join(p['name'] for p in critical) or 'none'}). Stable: {len(stable)}.",
        f"Beds: {beds_avail}/{total_beds} available. Ventilators: {vents_avail}/{total_vents} available.",
    ]
    if vitals_lines:
        lines.append("Vitals: " + " | ".join(vitals_lines))
    if siem_lines:
        lines.append("Recent security events: " + "; ".join(siem_lines))

    return " ".join(lines)


# ── System prompt — forces clean conversational output ────────────────────────
def _build_system_prompt() -> str:
    icu_context = _build_icu_context()
    return f"""You are a smart AI assistant embedded in a hospital ICU Digital Twin system.

IMPORTANT FORMATTING RULES — follow these strictly:
- Write in plain, natural sentences. No markdown tables. No raw asterisks. No pipe characters.
- Use bullet points (•) only when listing 3+ items. Keep them short.
- Bold words are fine when rendered (use **word**) but avoid overusing them.
- Be concise. Give complete answers in as few words as possible.
- For medical questions: be clinically precise and actionable.
- For general questions: be helpful and direct.

You have access to live ICU data. Use it when relevant. Never invent patient values.

LIVE ICU DATA: {icu_context}"""


# ── OpenRouter call — tries models in speed order ─────────────────────────────
async def _ask_llm(question: str, session_id: str) -> tuple[str, str]:
    """Returns (answer, model_used)"""
    if not OPENROUTER_API_KEY:
        return _fallback_engine(question), "fallback-keyword-engine"

    system_prompt = _build_system_prompt()
    history       = chat_sessions.get(session_id, [])

    messages = [{"role": "system", "content": system_prompt}]
    for entry in history[-MAX_HISTORY:]:
        messages.append({"role": "user",      "content": entry["question"]})
        messages.append({"role": "assistant",  "content": entry["answer"]})
    messages.append({"role": "user", "content": question})

    async with httpx.AsyncClient(timeout=30) as client:
        for model in FREE_MODELS:
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type":  "application/json",
                        "HTTP-Referer":  "https://capstone.dpdns.org",
                        "X-Title":       "ICU Digital Twin",
                    },
                    json={
                        "model":       model,
                        "messages":    messages,
                        "max_tokens":  500,
                        "temperature": 0.5,
                    },
                )

                data = response.json()

                if "choices" in data:
                    answer = data["choices"][0]["message"]["content"].strip()
                    print(f"✓ Responded: {model}")
                    return answer, model

                error = data.get("error", {})
                code  = error.get("code") if isinstance(error, dict) else None
                print(f"✗ {model} returned error {code} — trying next")

                if code not in (429, 503, 404):
                    # Unexpected error — still try next model
                    continue

            except Exception as e:
                print(f"✗ {model} exception: {e} — trying next")
                continue

    return _fallback_engine(question), "fallback-keyword-engine"


# ── Fallback keyword engine ───────────────────────────────────────────────────
def _fallback_engine(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["bed", "beds", "capacity"]):
        beds      = [r for r in fake_resources_db.values() if r["type"] == "bed"]
        available = sum(1 for b in beds if b["status"] == "available")
        return f"There are {len(beds)} ICU beds — {available} available and {len(beds)-available} occupied."

    if "ventilator" in q:
        vents     = [r for r in fake_resources_db.values() if r["type"] == "ventilator"]
        available = sum(1 for v in vents if v["status"] == "available")
        return f"There are {len(vents)} ventilators — {available} are currently available."

    if any(w in q for w in ["critical", "deteriorat", "at risk", "danger"]):
        critical = [p for p in fake_patients_db.values() if p["status"] == "critical"]
        if critical:
            return f"{len(critical)} critical patient(s) right now: {', '.join(p['name'] for p in critical)}."
        return "No patients are currently flagged as critical."

    if "sepsis" in q or "risk" in q:
        high_risk = [
            p["name"] for pid, p in fake_patients_db.items()
            if fake_vitals_db.get(pid) and (
                fake_vitals_db[pid][-1].get("spo2", 100) < 92
                or fake_vitals_db[pid][-1].get("blood_pressure_sys", 120) < 90
            )
        ]
        if high_risk:
            return f"High sepsis risk detected for: {', '.join(high_risk)}. Immediate clinical review recommended."
        return "No patients currently flagged as high sepsis risk."

    if any(w in q for w in ["summary", "status", "overview", "how many"]):
        beds_avail  = sum(1 for r in fake_resources_db.values() if r["type"] == "bed"        and r["status"] == "available")
        vents_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "ventilator" and r["status"] == "available")
        return (
            f"ICU has {len(fake_patients_db)} patients, "
            f"{beds_avail} beds available, and {vents_avail} ventilators available."
        )

    return (
        "AI model is currently unavailable. "
        "I can still answer questions about: beds, ventilators, critical patients, vitals, or ICU summary. "
        "Make sure OPENROUTER_API_KEY is set in Render environment variables."
    )


# ── POST /chatbot/query ───────────────────────────────────────────────────────
@router.post("/query", summary="Ask the AI assistant anything")
async def chatbot_query(body: ChatQuery, current_user=Depends(get_current_user)):
    """
    Ask anything — ICU data, medical questions, general knowledge.

    **Sample Request:**
    ```json
    {
      "question": "Provide deep analysis for Khalid and what are the recommendations",
      "session_id": "dr_ahmad_session"
    }
    ```

    **Sample Response:**
    ```json
    {
      "question": "Provide deep analysis for Khalid...",
      "answer": "Khalid Al-Mansouri is in critical condition...",
      "model": "z-ai/glm-4.5-air:free",
      "session_id": "dr_ahmad_session"
    }
    ```
    """
    answer, model_used = await _ask_llm(body.question, body.session_id)

    if body.session_id not in chat_sessions:
        chat_sessions[body.session_id] = []

    entry = {
        "question":  body.question,
        "answer":    answer,
        "timestamp": datetime.utcnow().isoformat(),
        "asked_by":  current_user["username"],
    }
    chat_sessions[body.session_id].append(entry)

    # Trim old history
    if len(chat_sessions[body.session_id]) > MAX_HISTORY * 2:
        chat_sessions[body.session_id] = chat_sessions[body.session_id][-MAX_HISTORY:]

    return {
        **entry,
        "session_id": body.session_id,
        "model":      model_used,
    }


# ── GET /chatbot/history ──────────────────────────────────────────────────────
@router.get("/history", summary="Get chat session history")
def get_history(session_id: str = "default", current_user=Depends(get_current_user)):
    history = chat_sessions.get(session_id, [])
    return {"session_id": session_id, "total_messages": len(history), "history": history}


# ── DELETE /chatbot/history ───────────────────────────────────────────────────
@router.delete("/history", summary="Clear chatbot session history")
def clear_history(session_id: str = "default", current_user=Depends(get_current_user)):
    chat_sessions.pop(session_id, None)
    return {"message": "Session cleared.", "session_id": session_id, "cleared_by": current_user["username"]}
