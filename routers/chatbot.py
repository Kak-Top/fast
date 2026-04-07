"""
Chatbot Router — Smart Output + Fast Routing
---------------------------------------------
Key improvements:
  1. Query classifier: deep analysis → best model, simple → fastest
  2. Server-side markdown→HTML conversion so the frontend always gets clean HTML
  3. System prompt with concrete good/bad examples the model actually follows
  4. Graceful fallback chain

Install: pip install httpx markdown2
Env:     OPENROUTER_API_KEY=your_key_here
"""

import os
import re
import httpx
import markdown2
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


class ChatQuery(BaseModel):
    question:   str
    session_id: str = "default"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# ── Model tiers ───────────────────────────────────────────────────────────────
# DEEP tier: used for analysis, recommendations, explanations — quality first
DEEP_MODELS = [
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "minimax/minimax-m2.5:free",
    "openrouter/free",
]

# FAST tier: used for simple lookups, summaries, yes/no answers
FAST_MODELS = [
    "z-ai/glm-4.5-air:free",
    "openai/gpt-oss-20b:free",
    "minimax/minimax-m2.5:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",
]

MAX_HISTORY = 6

# ── Keywords that trigger deep-analysis routing ───────────────────────────────
DEEP_KEYWORDS = {
    "analysis", "analyse", "analyze", "recommend", "recommendation",
    "explain", "differential", "diagnosis", "diagnose", "assessment",
    "prognosis", "treatment", "plan", "why", "interpret", "evaluate",
    "concern", "risk", "deteriorat", "suggest", "advice", "compare",
}


def _is_deep_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DEEP_KEYWORDS)


# ── ICU context ───────────────────────────────────────────────────────────────
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
                f"{p['name']} (Bed {p['bed_id']}, {p['status'].upper()}): "
                f"HR {v.get('heart_rate')} bpm | SpO2 {v.get('spo2')}% | "
                f"BP {v.get('blood_pressure_sys')}/{v.get('blood_pressure_dia')} mmHg | "
                f"Temp {v.get('temperature')}°C | RR {v.get('respiratory_rate')}/min"
            )

    beds_avail  = sum(1 for r in resources if r["type"] == "bed"        and r["status"] == "available")
    vents_avail = sum(1 for r in resources if r["type"] == "ventilator" and r["status"] == "available")
    total_beds  = sum(1 for r in resources if r["type"] == "bed")
    total_vents = sum(1 for r in resources if r["type"] == "ventilator")

    recent_siem = fake_siem_events_db[-3:] if fake_siem_events_db else []
    siem_lines  = [
        f"[{e.get('severity')}] {e.get('event_type')}: {e.get('description')}"
        for e in recent_siem
    ]

    lines = [
        f"Patients: {len(patients)} total | Critical: {len(critical)} ({', '.join(p['name'] for p in critical) or 'none'}) | Stable: {len(stable)}",
        f"Resources: {beds_avail}/{total_beds} beds free | {vents_avail}/{total_vents} ventilators free",
    ]
    if vitals_lines:
        lines.append("Latest vitals: " + " || ".join(vitals_lines))
    if siem_lines:
        lines.append("Recent security events: " + " | ".join(siem_lines))

    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt(deep: bool = False) -> str:
    icu_context = _build_icu_context()
    depth_note = (
        "The user is asking for a deep clinical analysis. Be thorough, structured, and precise."
        if deep else
        "Keep the response brief and direct."
    )

    return f"""You are an expert AI clinical assistant embedded in a hospital ICU Digital Twin system.

{depth_note}

FORMATTING RULES — follow exactly, no exceptions:
- Use Markdown formatting: **bold** for key terms, ## for section headers, - for bullet points.
- For clinical data, use structured sections with headers like: ## Patient Overview, ## Vital Signs, ## Clinical Concerns, ## Recommendations.
- Present vitals as bullet points, NOT as pipe tables. Example:
  - **Heart Rate**: 118 bpm — Tachycardia, compensatory for low BP
  - **SpO2**: 89% — Hypoxemia, needs supplemental O2
- For recommendations, number them: 1. 2. 3.
- Never output raw pipe/table syntax like: | HR | Value | Range |
- Never output triple-dashes (---) or equals signs (===) as dividers.
- Be clinically accurate. Never invent patient values.

LIVE ICU DATA:
{icu_context}"""


# ── Markdown → clean HTML ─────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """
    Convert markdown to HTML.
    Also strip any accidental pipe-table lines that slip through.
    """
    # Remove raw pipe-table rows the model sometimes still outputs
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        # Skip markdown table separator rows (|---|---|) and table rows
        stripped = line.strip()
        if re.match(r"^\|[-| :]+\|$", stripped):
            continue
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3:
            # Convert table row to bullet point instead of dropping it
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            cells = [c for c in cells if c]
            if cells:
                clean_lines.append("- " + " — ".join(cells))
            continue
        clean_lines.append(line)

    cleaned_md = "\n".join(clean_lines)

    # Convert markdown to HTML using markdown2 with extras
    html = markdown2.markdown(
        cleaned_md,
        extras=["fenced-code-blocks", "strike", "tables", "cuddled-lists", "break-on-newline"],
    )
    return html.strip()


# ── OpenRouter call ───────────────────────────────────────────────────────────
async def _ask_llm(question: str, session_id: str) -> tuple[str, str]:
    """Returns (html_answer, model_used)"""
    if not OPENROUTER_API_KEY:
        return _fallback_engine(question), "fallback-keyword-engine"

    deep      = _is_deep_query(question)
    models    = DEEP_MODELS if deep else FAST_MODELS
    system_p  = _build_system_prompt(deep=deep)
    history   = chat_sessions.get(session_id, [])

    messages = [{"role": "system", "content": system_p}]
    for entry in history[-MAX_HISTORY:]:
        messages.append({"role": "user",      "content": entry["question"]})
        messages.append({"role": "assistant",  "content": entry["answer_raw"]})
    messages.append({"role": "user", "content": question})

    async with httpx.AsyncClient(timeout=40) as client:
        for model in models:
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
                        "max_tokens":  800 if deep else 400,
                        "temperature": 0.4,
                    },
                )

                data = response.json()

                if "choices" in data:
                    raw    = data["choices"][0]["message"]["content"].strip()
                    html   = _md_to_html(raw)
                    print(f"✓ [{('DEEP' if deep else 'FAST')}] Responded: {model}")
                    return html, raw, model

                error = data.get("error", {})
                code  = error.get("code") if isinstance(error, dict) else None
                print(f"✗ {model} → error {code}, trying next")

            except Exception as e:
                print(f"✗ {model} → exception: {e}, trying next")
                continue

    raw = _fallback_engine(question)
    return _md_to_html(raw), raw, "fallback-keyword-engine"


# ── Fallback keyword engine ───────────────────────────────────────────────────
def _fallback_engine(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["bed", "beds", "capacity"]):
        beds      = [r for r in fake_resources_db.values() if r["type"] == "bed"]
        available = sum(1 for b in beds if b["status"] == "available")
        return f"**Bed Status**: {available} of {len(beds)} ICU beds are currently available ({len(beds)-available} occupied)."

    if "ventilator" in q:
        vents     = [r for r in fake_resources_db.values() if r["type"] == "ventilator"]
        available = sum(1 for v in vents if v["status"] == "available")
        return f"**Ventilators**: {available} of {len(vents)} are currently available."

    if any(w in q for w in ["critical", "deteriorat", "at risk", "danger"]):
        critical = [p for p in fake_patients_db.values() if p["status"] == "critical"]
        if critical:
            names = ", ".join(p["name"] for p in critical)
            return f"**{len(critical)} critical patient(s)** currently: {names}."
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
            return f"**High sepsis risk**: {', '.join(high_risk)}. Immediate clinical review recommended."
        return "No patients currently flagged as high sepsis risk."

    if any(w in q for w in ["summary", "status", "overview", "how many"]):
        beds_avail  = sum(1 for r in fake_resources_db.values() if r["type"] == "bed"        and r["status"] == "available")
        vents_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "ventilator" and r["status"] == "available")
        return (
            f"**ICU Overview**: {len(fake_patients_db)} patients | "
            f"{beds_avail} beds free | {vents_avail} ventilators free."
        )

    return (
        "**AI model unavailable.** I can still answer questions about: "
        "beds, ventilators, critical patients, vitals, or ICU summary. "
        "Ensure `OPENROUTER_API_KEY` is set in Render environment variables."
    )


# ── POST /chatbot/query ───────────────────────────────────────────────────────
@router.post("/query", summary="Ask the AI assistant anything")
async def chatbot_query(body: ChatQuery, current_user=Depends(get_current_user)):
    """
    Ask anything — ICU data, medical questions, general knowledge.

    Returns HTML-formatted answer ready to render in the frontend.

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
      "question": "...",
      "answer": "<h2>Patient Overview</h2><p>Khalid Al-Mansouri...</p>",
      "answer_raw": "## Patient Overview\\nKhalid...",
      "model": "openai/gpt-oss-120b:free",
      "session_id": "dr_ahmad_session",
      "query_type": "deep"
    }
    ```
    """
    result = await _ask_llm(body.question, body.session_id)
    answer_html, answer_raw, model_used = result

    if body.session_id not in chat_sessions:
        chat_sessions[body.session_id] = []

    entry = {
        "question":   body.question,
        "answer":     answer_html,
        "answer_raw": answer_raw,
        "timestamp":  datetime.utcnow().isoformat(),
        "asked_by":   current_user["username"],
    }
    chat_sessions[body.session_id].append(entry)

    if len(chat_sessions[body.session_id]) > MAX_HISTORY * 2:
        chat_sessions[body.session_id] = chat_sessions[body.session_id][-MAX_HISTORY:]

    return {
        "question":   body.question,
        "answer":     answer_html,
        "session_id": body.session_id,
        "model":      model_used,
        "query_type": "deep" if _is_deep_query(body.question) else "fast",
        "timestamp":  entry["timestamp"],
        "asked_by":   entry["asked_by"],
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
