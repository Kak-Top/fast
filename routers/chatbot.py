"""
Chatbot Router — Smart Output + Fast Routing + Code Highlighting + Table Auto-Fix
---------------------------------------------------------------------------------
Key improvements:
  1. Query classifier: code/medical/table/general → best model tier
  2. Auto syntax highlighting: detects language → adds colored code blocks
  3. Auto-fix markdown tables: inserts missing separator rows before HTML conversion
  4. Server-side markdown→HTML with styled tables, headers, lists, and code
  5. New model tiers: Qwen3, Nemotron, Gemma, Step-3.5 prioritized for quality
  6. Graceful fallback chain + intelligent model selection

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
CODE_MODELS = [
    "qwen/qwen3-coder:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "stepfun/step-3.5-flash:free",
]

DEEP_MODELS = [
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-120b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "stepfun/step-3.5-flash:free",
    "minimax/minimax-m2.5:free",
    "openrouter/free",
]

FAST_MODELS = [
    "qwen/qwen3-coder:free",
    "stepfun/step-3.5-flash:free",
    "qwen/qwen-2.5-32b-instruct:free",
    "z-ai/glm-4.5-air:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "minimax/minimax-m2.5:free",
    "openrouter/free",
]

MAX_HISTORY = 6

# ── Keywords for query classification ─────────────────────────────────────────
DEEP_KEYWORDS = {
    "analysis", "analyse", "analyze", "recommend", "recommendation",
    "explain", "differential", "diagnosis", "diagnose", "assessment",
    "prognosis", "treatment", "plan", "why", "interpret", "evaluate",
    "concern", "risk", "deteriorat", "suggest", "advice", "compare",
    "clinical", "patient", "vitals", "sepsis", "shock", "icu",
}

TABLE_KEYWORDS = {
    "table", "compare", "vs", "versus", "all patients", "list all",
    "grid", "side by side", "tabulate", "show me all", "matrix",
}

CODE_KEYWORDS = {
    "code", "python", "rust", "javascript", "java", "cpp", "c++", "c#",
    "function", "debug", "error", "fix", "implement", "write", "script",
    "api", "endpoint", "request", "response", "json", "xml", "sql",
    "class", "struct", "enum", "trait", "impl", "fn", "pub", "async",
}


def _classify_query(question: str) -> str:
    """
    Classify query into: 'code', 'deep', 'table', or 'fast'
    Returns the best category for model routing.
    """
    q = question.lower()

    if any(kw in q for kw in CODE_KEYWORDS):
        return "code"

    if any(kw in q for kw in TABLE_KEYWORDS):
        return "table"

    if any(kw in q for kw in DEEP_KEYWORDS):
        return "deep"

    return "fast"


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
        f"Patients: {len(patients)} total | Critical: {len(critical)} "
        f"({', '.join(p['name'] for p in critical) or 'none'}) | Stable: {len(stable)}",
        f"Resources: {beds_avail}/{total_beds} beds free | {vents_avail}/{total_vents} ventilators free",
    ]
    if vitals_lines:
        lines.append("Latest vitals: " + " || ".join(vitals_lines))
    if siem_lines:
        lines.append("Recent security events: " + " | ".join(siem_lines))

    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────
def _build_system_prompt(query_type: str) -> str:
    icu_context = _build_icu_context()

    if query_type == "code":
        depth_note = """You are helping with code. Provide clean, well-commented examples.
- Use fenced code blocks with language specifier: ```rust, ```python, etc.
- Explain logic clearly before/after code.
- Include error handling and best practices."""
    elif query_type == "table":
        depth_note = """Present comparisons or lists in markdown tables.
- ALWAYS include the separator row after headers: |---|---|---|
- Keep tables concise but informative."""
    elif query_type == "deep":
        depth_note = """Provide a thorough, structured clinical analysis.
- Use bullet points for vitals, numbered lists for recommendations.
- Be precise, evidence-based, and actionable."""
    else:
        depth_note = "Keep responses brief, direct, and factual."

    return f"""You are an expert AI clinical assistant embedded in a hospital ICU Digital Twin system.
You can also help with code, general questions, and data formatting.

{depth_note}

FORMATTING RULES — follow exactly:

**For Code Blocks:**
- Always use fenced code blocks with language: ```rust, ```python, ```javascript
- Example:
  ```rust
  fn main() {{
      println!("Hello, ICU!");
  }}
  ```

**For Individual Patient Vitals:**
- Use bullet points:
  - **Heart Rate**: 118 bpm — Tachycardia, compensatory for low BP
  - **SpO2**: 89% — Hypoxemia, needs supplemental O₂

**For Comparisons / Multiple Patients / Lists:**
- Use markdown tables WITH SEPARATOR ROWS:
  | Patient | Bed | Status | HR | SpO2 | BP |
  |---------|-----|--------|-----|------|-----|
  | Khalid  | 3A  | Critical | 118 | 89% | 85/52 |
  ⚠️ The separator row (|---|) is REQUIRED for tables to render!

**For Recommendations:**
- Number them: 1. 2. 3.
- Use **bold** for critical actions or terms

**Section Headers:**
- Use ## for main sections: ## Patient Overview, ## Vital Signs, ## Recommendations

**NEVER:**
- Use triple-dashes (---) or equals signs (===) as standalone dividers
- Output raw HTML tags like <table>, <tr>, <pre>
- Invent patient data not in the LIVE ICU DATA section

LIVE ICU DATA (use this for answers):
{icu_context}"""


# ── Language detection for code blocks ────────────────────────────────────────
def _detect_language(code_block: str) -> str:
    """
    Detect programming language from code content using heuristics.
    Returns lowercase language name for syntax highlighting.
    """
    code = code_block.strip()
    code_lower = code.lower()

    if re.search(r'\b(fn|let\s+mut|impl\s+\w+|trait\s+\w+|pub\s+fn|unsafe\s*\{)\b', code):
        return "rust"
    if code_lower.startswith("use ") and re.search(r'\b(std::|crate::)\b', code):
        return "rust"

    if re.search(r'\b(def\s+\w+|import\s+\w+|from\s+\w+\s+import|class\s+\w+:|if\s+__name__)\b', code):
        return "python"

    if re.search(r'\b(const|let|var)\s+\w+\s*=|=>|async\s+function|import\s+\{', code):
        return "javascript"

    if re.search(r'\bpublic\s+class\s+\w+|void\s+main|using\s+System;', code):
        return "java" if "package " in code_lower else "csharp"

    if re.search(r'#include\s*[<"]|int\s+main\s*\(|std::', code):
        return "cpp" if "std::" in code else "c"

    if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE|FROM|WHERE|JOIN)\b', code.upper()):
        return "sql"

    if code.startswith("#!") or re.search(r'\b(bash|sh)\b|echo \$|grep -', code_lower):
        return "bash"

    if code.strip().startswith(("{", "[")) and re.search(r'"\w+"\s*:', code):
        return "json"

    return "plaintext"


# ── Syntax highlighting styles ────────────────────────────────────────────────
_DEFAULT_STYLE = {
    "keyword": "#c586c0",
    "function": "#dcdcaa",
    "string": "#ce9178",
    "comment": "#6a9955",
    "number": "#b5cea8",
    "type": "#4ec9b0",
    "background": "#1e1e1e",
    "text": "#d4d4d4",
}

_CODE_STYLES = {
    "rust": _DEFAULT_STYLE,
    "python": {**_DEFAULT_STYLE, "keyword": "#569cd6"},
    "javascript": {**_DEFAULT_STYLE, "keyword": "#569cd6"},
    "sql": {**_DEFAULT_STYLE, "keyword": "#569cd6"},
    "plaintext": _DEFAULT_STYLE,
}


def _apply_syntax_highlighting(code: str, language: str) -> str:
    """
    Apply inline CSS syntax highlighting to a code block.
    Returns a complete styled <pre><code> HTML block.
    """
    styles = _CODE_STYLES.get(language, _DEFAULT_STYLE)

    lines = code.split('\n')
    highlighted_lines = []

    for line in lines:
        stripped = line.strip()

        # Full-line comments
        if stripped.startswith('#') or stripped.startswith('//'):
            highlighted_lines.append(
                f'<span style="color:{styles["comment"]}">{line}</span>'
            )
            continue

        # Strings
        line = re.sub(
            r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
            f'<span style="color:{styles["string"]}">\\1</span>',
            line,
        )

        # Keywords
        keywords = (
            r'\b(if|else|elif|for|while|return|fn|pub|struct|enum|trait|impl|'
            r'def|class|import|from|async|await|const|let|var|function|'
            r'SELECT|FROM|WHERE|JOIN|INSERT|UPDATE|DELETE|use|mod|match|'
            r'true|false|null|None|True|False)\b'
        )
        line = re.sub(
            keywords,
            f'<span style="color:{styles["keyword"]};font-weight:600">\\1</span>',
            line,
            flags=re.IGNORECASE,
        )

        # Function calls
        line = re.sub(
            r'(\b\w+)(\s*\()',
            f'<span style="color:{styles["function"]}">\\1</span>\\2',
            line,
        )

        # Numbers
        line = re.sub(
            r'\b(\d+\.?\d*)\b',
            f'<span style="color:{styles["number"]}">\\1</span>',
            line,
        )

        # Types (PascalCase words)
        line = re.sub(
            r'\b([A-Z][a-zA-Z0-9_]*)\b',
            f'<span style="color:{styles["type"]}">\\1</span>',
            line,
        )

        highlighted_lines.append(line)

    highlighted_code = '\n'.join(highlighted_lines)

    return (
        f'<pre style="background:{styles["background"]};color:{styles["text"]};'
        f'padding:14px 18px;border-radius:8px;overflow-x:auto;'
        f'font-family:\'Fira Code\',\'Consolas\',monospace;font-size:0.9em;line-height:1.5">'
        f'<code>{highlighted_code}</code></pre>'
    )


# ── Auto-fix malformed markdown tables ────────────────────────────────────────
def _fix_markdown_tables(text: str) -> str:
    """
    Detect markdown tables missing separator rows and auto-insert them.
    """
    lines = text.splitlines()
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        is_table_row = (
            stripped.startswith('|')
            and stripped.endswith('|')
            and stripped.count('|') >= 3
        )

        if is_table_row and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()

            is_separator = bool(
                next_stripped.startswith('|')
                and next_stripped.endswith('|')
                and re.match(r'^\|[\s\-:|]+\|$', next_stripped)
                and next_stripped.count('-') >= (next_stripped.count('|') - 1)
            )

            if not is_separator:
                cells = [c.strip() for c in stripped.strip('|').split('|') if c.strip()]
                if len(cells) >= 2:
                    separator = '| ' + ' | '.join(['---'] * len(cells)) + ' |'
                    fixed_lines.append(line)
                    fixed_lines.append(separator)
                    i += 1
                    continue

        fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines)


# ── Markdown → styled HTML ────────────────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """
    Convert markdown to HTML with:
    - Auto-fixed tables
    - Syntax-highlighted fenced code blocks
    - Styled tables, headers, and lists
    """
    # Step 1: Fix malformed tables
    text = _fix_markdown_tables(text)

    # Step 2: Replace fenced code blocks with highlighted HTML BEFORE markdown2
    def replace_code_block(match: re.Match) -> str:
        lang = match.group(1).strip().lower()
        code = match.group(2)
        if not lang or lang == "plaintext":
            lang = _detect_language(code)
        return replace_code_block.placeholder + f"CODEBLOCK_{len(replace_code_block.blocks)}_END"

    # We store rendered blocks and substitute placeholders to avoid markdown2
    # double-processing the highlighted HTML
    code_blocks: list[str] = []

    def extract_code_block(match: re.Match) -> str:
        lang = match.group(1).strip().lower()
        code = match.group(2)
        if not lang or lang == "plaintext":
            lang = _detect_language(code)
        rendered = _apply_syntax_highlighting(code, lang)
        idx = len(code_blocks)
        code_blocks.append(rendered)
        return f"\n\nCODEPLACEHOLDER_{idx}\n\n"

    text = re.sub(r'```(\w*)\n(.*?)```', extract_code_block, text, flags=re.DOTALL)

    # Step 3: markdown2 conversion
    html = markdown2.markdown(
        text,
        extras=[
            "tables",
            "strike",
            "cuddled-lists",
            "break-on-newline",
            "header-ids",
        ],
    )

    # Step 4: Restore highlighted code blocks
    for idx, block in enumerate(code_blocks):
        html = html.replace(f"<p>CODEPLACEHOLDER_{idx}</p>", block)
        html = html.replace(f"CODEPLACEHOLDER_{idx}", block)

    # Step 5: Style tables
    if "<table>" in html:
        html = html.replace(
            "<table>",
            '<table style="border-collapse:collapse;width:100%;margin:1em 0;'
            'font-size:0.95em;border-radius:6px;overflow:hidden;">',
        )
        html = html.replace(
            "<th>",
            '<th style="border:1px solid #444;padding:10px 12px;'
            'background:linear-gradient(135deg,#1e3a5f,#2d5a87);'
            'color:white;text-align:left;font-weight:600;">',
        )
        html = html.replace(
            "<td>",
            '<td style="border:1px solid #ddd;padding:10px 12px;">',
        )
        # Zebra striping on <tbody> rows
        html = re.sub(
            r'(<tbody>\s*)(<tr>)',
            r'\1<tr style="background-color:#f9f9f9;">',
            html,
            count=1,
        )

    # Step 6: Style h2 headers
    html = re.sub(
        r'<h2(.*?)>(.*?)</h2>',
        r'<h2\1 style="color:#1e3a5f;border-bottom:2px solid #2d5a87;'
        r'padding-bottom:8px;margin-top:1.5em;margin-bottom:1em;">\2</h2>',
        html,
    )

    # Step 7: Style list items that start with bold (clinical vitals pattern)
    html = re.sub(
        r'<li>(<strong>[^<]+</strong>[^<]*)</li>',
        r'<li style="margin:6px 0;padding-left:8px;line-height:1.6;">\1</li>',
        html,
    )

    # Step 8: Fallback styling for any remaining bare <pre><code> blocks
    html = html.replace(
        "<pre><code>",
        '<pre style="background:#1e293b;color:#e2e8f0;padding:14px 18px;'
        'border-radius:8px;overflow-x:auto;font-family:monospace;font-size:0.9em"><code>',
    )

    return html.strip()


# ── Fallback keyword engine ───────────────────────────────────────────────────
def _fallback_engine(question: str) -> str:
    q = question.lower()

    if any(w in q for w in ["bed", "beds", "capacity"]):
        beds = [r for r in fake_resources_db.values() if r["type"] == "bed"]
        available = sum(1 for b in beds if b["status"] == "available")
        return (
            f"**Bed Status**: {available} of {len(beds)} ICU beds are currently available "
            f"({len(beds) - available} occupied)."
        )

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
            p["name"]
            for pid, p in fake_patients_db.items()
            if fake_vitals_db.get(pid)
            and (
                fake_vitals_db[pid][-1].get("spo2", 100) < 92
                or fake_vitals_db[pid][-1].get("blood_pressure_sys", 120) < 90
            )
        ]
        if high_risk:
            return (
                f"**High sepsis risk**: {', '.join(high_risk)}. "
                "Immediate clinical review recommended."
            )
        return "No patients currently flagged as high sepsis risk."

    if any(w in q for w in ["summary", "status", "overview", "how many"]):
        beds_avail = sum(
            1 for r in fake_resources_db.values()
            if r["type"] == "bed" and r["status"] == "available"
        )
        vents_avail = sum(
            1 for r in fake_resources_db.values()
            if r["type"] == "ventilator" and r["status"] == "available"
        )
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

    query_type = _classify_query(question)

    if query_type == "code":
        models = CODE_MODELS
        max_tokens = 1500
        temperature = 0.2
    elif query_type in ("deep", "table"):
        models = DEEP_MODELS
        max_tokens = 1200
        temperature = 0.3
    else:
        models = FAST_MODELS
        max_tokens = 500
        temperature = 0.5

    system_prompt = _build_system_prompt(query_type)
    history = chat_sessions.get(session_id, [])

    messages = [{"role": "system", "content": system_prompt}]
    for entry in history[-MAX_HISTORY:]:
        messages.append({"role": "user", "content": entry["question"]})
        messages.append({"role": "assistant", "content": entry["answer_raw"]})
    messages.append({"role": "user", "content": question})

    async with httpx.AsyncClient(timeout=90) as client:
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

                if (
                    "choices" in data
                    and data["choices"]
                    and "message" in data["choices"][0]
                ):
                    raw = data["choices"][0]["message"]["content"].strip()
                    html = _md_to_html(raw)
                    print(f"✓ [{query_type.upper()}] Model: {model}")
                    return html, raw, model

                error = data.get("error", {})
                err_msg = (
                    error.get("message", "unknown") if isinstance(error, dict) else str(error)
                )
                print(f"✗ {model} → {err_msg}, trying next")

            except httpx.TimeoutException:
                print(f"✗ {model} → timeout, trying next")
            except Exception as e:
                print(f"✗ {model} → {type(e).__name__}: {e}, trying next")

    print("⚠️ All models failed, using fallback engine")
    raw = _fallback_engine(question)
    return _md_to_html(raw), raw, "fallback-keyword-engine"


# ── POST /chatbot/query ───────────────────────────────────────────────────────
@router.post("/query", summary="Ask the AI assistant anything")
async def chatbot_query(body: ChatQuery, current_user=Depends(get_current_user)):
    """
    Ask anything — ICU data, medical questions, code help, general knowledge.
    Returns HTML-formatted answer ready to render in the frontend.
    """
    answer_html, answer_raw, model_used = await _ask_llm(body.question, body.session_id)

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

    query_type = _classify_query(body.question)

    return {
        "question": body.question,
        "answer": answer_html,
        "answer_raw": answer_raw,
        "session_id": body.session_id,
        "model": model_used,
        "query_type": query_type,
        "timestamp": entry["timestamp"],
        "asked_by": entry["asked_by"],
    }


# ── GET /chatbot/history ──────────────────────────────────────────────────────
@router.get("/history", summary="Get chat session history")
def get_history(
    session_id: str = "default", current_user=Depends(get_current_user)
):
    history = chat_sessions.get(session_id, [])
    return {
        "session_id": session_id,
        "total_messages": len(history),
        "history": history,
    }


# ── DELETE /chatbot/history ───────────────────────────────────────────────────
@router.delete("/history", summary="Clear chatbot session history")
def clear_history(
    session_id: str = "default", current_user=Depends(get_current_user)
):
    chat_sessions.pop(session_id, None)
    return {
        "message": "Session cleared.",
        "session_id": session_id,
        "cleared_by": current_user["username"],
    }
