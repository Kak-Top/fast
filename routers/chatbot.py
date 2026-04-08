"""
Chatbot Router — Smart Output + Fast Routing + Table Support
-------------------------------------------------------------
Key improvements:
  1. Query classifier: deep analysis → best model, simple → fastest
  2. Smart table detection: user asks for table → model creates table
  3. Server-side markdown→HTML with proper table styling
  4. System prompt with flexible formatting rules (tables allowed!)
  5. Graceful fallback chain + Qwen models prioritized

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
    question: str
    session_id: str = "default"


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Model tiers ───────────────────────────────────────────────────────────────
# DEEP tier: Best models for analysis, tables, recommendations
DEEP_MODELS = [
    "qwen/qwen-2.5-72b-instruct:free",      # Excellent for medical + structured output
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "minimax/minimax-m2.5:free",
    "openrouter/free",
]

# FAST tier: Quick responses for simple queries
FAST_MODELS = [
    "qwen/qwen-2.5-32b-instruct:free",      # Fast + smart
    "z-ai/glm-4.5-air:free",
    "openai/gpt-oss-20b:free",
    "minimax/minimax-m2.5:free",
    "openrouter/free",
]

MAX_HISTORY = 6

# ── Keywords ─────────────────────────────────────────────────────────────────
DEEP_KEYWORDS = {
    "analysis", "analyse", "analyze", "recommend", "recommendation",
    "explain", "differential", "diagnosis", "diagnose", "assessment",
    "prognosis", "treatment", "plan", "why", "interpret", "evaluate",
    "concern", "risk", "deteriorat", "suggest", "advice", "compare",
}

TABLE_KEYWORDS = {
    "table", "compare", "vs", "versus", "all patients", "list all", 
    "grid", "side by side", "tabulate", "show me all"
}


def _is_deep_query(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in DEEP_KEYWORDS)


def _wants_table(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in TABLE_KEYWORDS)


# ── ICU context ───────────────────────────────────────────────────────────────
def _build_icu_context() -> str:
    patients = list(fake_patients_db.values())
    resources = list(fake_resources_db.values())

    critical = [p for p in patients if p.get("status") == "critical"]
    stable = [p for p in patients if p.get("status") == "stable"]

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

    beds_avail = sum(1 for r in resources if r["type"] == "bed" and r["status"] == "available")
    vents_avail = sum(1 for r in resources if r["type"] == "ventilator" and r["status"] == "available")
    total_beds = sum(1 for r in resources if r["type"] == "bed")
    total_vents = sum(1 for r in resources if r["type"] == "ventilator")

    recent_siem = fake_siem_events_db[-3:] if fake_siem_events_db else []
    siem_lines = [
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
def _build_system_prompt(deep: bool = False, wants_table: bool = False) -> str:
    icu_context = _build_icu_context()
    
    if deep and wants_table:
        depth_note = "Provide a thorough, structured clinical analysis. Use markdown tables when comparing patients or showing multiple data points."
    elif deep:
        depth_note = "Provide a thorough, structured clinical analysis. Use bullet points for vitals and numbered lists for recommendations."
    elif wants_table:
        depth_note = "Keep it concise but use a markdown table to present the requested comparison or list."
    else:
        depth_note = "Keep the response brief and direct."

    return f"""You are an expert AI clinical assistant embedded in a hospital ICU Digital Twin system.

{depth_note}

FORMATTING RULES — follow exactly:

**For Individual Patient Vitals:**
- Use bullet points:
  - **Heart Rate**: 118 bpm — Tachycardia, compensatory for low BP
  - **SpO2**: 89% — Hypoxemia, needs supplemental O₂
  - **BP**: 85/52 mmHg — Hypotensive

**For Comparisons / Multiple Patients / Lists:**
- Use markdown tables when the user asks for a table, comparison, or "all patients":
  | Patient | Bed | Status | HR | SpO2 | BP | Notes |
  |---------|-----|--------|-----|------|-----|-------|
  | Khalid  | 3A  | Critical | 118 | 89% | 85/52 | Sepsis risk |

**For Recommendations:**
- Number them clearly: 1. 2. 3.
- Use **bold** for critical actions or terms

**Section Headers:**
- Use ## for main sections: ## Patient Overview, ## Vital Signs, ## Clinical Concerns, ## Recommendations

**NEVER:**
- Use triple-dashes (---) or equals signs (===) as dividers
- Output raw HTML tags like <table> or <tr>
- Invent patient data not in the LIVE ICU DATA section

LIVE ICU DATA (use this for answers):
{icu_context}"""


# ── Markdown → clean HTML with table styling ──────────────────────────────────
def _md_to_html(text: str) -> str:
    """
    Convert markdown to HTML with proper table support and styling.
    """
    # Clean up any accidental malformed lines but preserve tables
    lines = text.splitlines()
    clean_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Skip only pure table separator rows that are malformed
        if re.match(r"^\|[\s\-:|]+\|$", stripped) and stripped.count("-") > len(stripped) * 0.5:
            continue
        clean_lines.append(line)
    
    cleaned_md = "\n".join(clean_lines)

    # Convert markdown to HTML
    html = markdown2.markdown(
        cleaned_md,
        extras=[
            "fenced-code-blocks",
            "tables",              # Critical for table support
            "strike",
            "cuddled-lists",
            "break-on-newline",
            "header-ids",
        ],
    )
    
    # Add CSS styling for tables to make them readable
    if "<table>" in html:
        html = html.replace(
            "<table>",
            '<table style="border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.95em;">'
        )
        html = html.replace(
            "<th>",
            '<th style="border: 1px solid #444; padding: 10px 12px; background: linear-gradient(135deg, #1e3a5f, #2d5a87); color: white; text-align: left; font-weight: 600;">'
        )
        html = html.replace(
            "<td>",
            '<td style="border: 1px solid #ddd; padding: 10px 12px;">'
        )
        # Add zebra striping for readability
        html = html.replace(
            "<tr>",
            '<tr style="background-color: #f9f9f9;">'
        ).replace(
            '<tr style="background-color: #f9f9f9;">',
            '<tr>',
            1  # Only replace first occurrence to keep header row styled
        )
    
    # Style headers for better hierarchy
    html = re.sub(
        r'<h2>(.*?)</h2>',
        r'<h2 style="color: #1e3a5f; border-bottom: 2px solid #2d5a87; padding-bottom: 8px; margin-top: 1.5em;">\1</h2>',
        html
    )
    
    # Style bullet points for clinical data
    html = re.sub(
        r'<li>(<strong>.*?</strong>:.*?)</li>',
        r'<li style="margin: 4px 0; padding-left: 8px;">\1</li>',
        html
    )
    
    return html.strip()


# ── Fallback keyword engine ───────────────────────────────────────────────────
def _fallback_engine(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["bed", "beds", "capacity"]):
        beds = [r for r in fake_resources_db.values() if r["type"] == "bed"]
        available = sum(1 for b in beds if b["status"] == "available")
        return f"**Bed Status**: {available} of {len(beds)} ICU beds are currently available ({len(beds)-available} occupied)."

    if "ventilator" in q:
        vents = [r for r in fake_resources_db.values() if r["type"] == "ventilator"]
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
        beds_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "bed" and r["status"] == "available")
        vents_avail = sum(1 for r in fake_resources_db.values() if r["type"] == "ventilator" and r["status"] == "available")
        return (
            f"**ICU Overview**: {len(fake_patients_db)} patients | "
            f"{beds_avail} beds free | {vents_avail} ventilators free."
        )

    return (
        "**⚠️ AI model unavailable.** I can still answer questions about: "
        "beds, ventilators, critical patients, vitals, or ICU summary. "
        "Ensure `OPENROUTER_API_KEY` is set in your environment variables."
    )


# ── OpenRouter call ───────────────────────────────────────────────────────────
async def _ask_llm(question: str, session_id: str) -> tuple[str, str, str]:
    """Returns (html_answer, raw_answer, model_used)"""
    
    if not OPENROUTER_API_KEY:
        raw = _fallback_engine(question)
        return _md_to_html(raw), raw, "fallback-keyword-engine"

    deep = _is_deep_query(question)
    wants_tbl = _wants_table(question)
    
    # Choose models and params based on query type
    if deep or wants_tbl:
        models = DEEP_MODELS
        max_tokens = 1200
        temperature = 0.3  # Lower = more precise for medical
    else:
        models = FAST_MODELS
        max_tokens = 500
        temperature = 0.5

    system_prompt = _build_system_prompt(deep=deep, wants_table=wants_tbl)
    history = chat_sessions.get(session_id, [])

    # Build messages with history
    messages = [{"role": "system", "content": system_prompt}]
    for entry in history[-MAX_HISTORY:]:
        messages.append({"role": "user", "content": entry["question"]})
        messages.append({"role": "assistant", "content": entry["answer_raw"]})
    messages.append({"role": "user", "content": question})

    async with httpx.AsyncClient(timeout=60) as client:
        for model in models:
            try:
                response = await client.post(
                    OPENROUTER_URL,
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://capstone.dpdns.org",
                        "X-Title": "ICU Digital Twin",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_p": 0.9,
                    },
                )

                data = response.json()

                if "choices" in data and data["choices"] and "message" in data["choices"][0]:
                    raw = data["choices"][0]["message"]["content"].strip()
                    html = _md_to_html(raw)
                    print(f"✓ [{'DEEP' if deep else 'FAST'}][{'TABLE' if wants_tbl else 'TEXT'}] Model: {model}")
                    return html, raw, model

                error = data.get("error", {})
                err_msg = error.get("message", "unknown") if isinstance(error, dict) else error
                print(f"✗ {model} → {err_msg}, trying next")

            except httpx.TimeoutException:
                print(f"✗ {model} → timeout, trying next")
                continue
            except Exception as e:
                print(f"✗ {model} → {type(e).__name__}: {e}, trying next")
                continue

    # All models failed → fallback
    print("⚠️ All models failed, using fallback engine")
    raw = _fallback_engine(question)
    return _md_to_html(raw), raw, "fallback-keyword-engine"


# ── POST /chatbot/query ───────────────────────────────────────────────────────
@router.post("/query", summary="Ask the AI assistant anything")
async def chatbot_query(body: ChatQuery, current_user=Depends(get_current_user)):
    """
    Ask anything — ICU data, medical questions, general knowledge.
    Returns HTML-formatted answer ready to render in the frontend.
    """
    answer_html, answer_raw, model_used = await _ask_llm(body.question, body.session_id)

    # Initialize session if needed
    if body.session_id not in chat_sessions:
        chat_sessions[body.session_id] = []

    entry = {
        "question": body.question,
        "answer": answer_html,
        "answer_raw": answer_raw,
        "timestamp": datetime.utcnow().isoformat(),
        "asked_by": current_user["username"],
        "model": model_used,
    }
    chat_sessions[body.session_id].append(entry)

    # Trim history to avoid memory bloat
    if len(chat_sessions[body.session_id]) > MAX_HISTORY * 2:
        chat_sessions[body.session_id] = chat_sessions[body.session_id][-MAX_HISTORY:]

    return {
        "question": body.question,
        "answer": answer_html,        # ✅ Ready-to-render HTML
        "answer_raw": answer_raw,     # For debugging if needed
        "session_id": body.session_id,
        "model": model_used,
        "query_type": "deep" if _is_deep_query(body.question) else "fast",
        "wants_table": _wants_table(body.question),
        "timestamp": entry["timestamp"],
        "asked_by": entry["asked_by"],
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
