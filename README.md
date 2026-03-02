# 🏥 ICU Digital Twin — FastAPI Backend

A complete FastAPI backend for the Hospital ICU Digital Twin simulation system,
built for a Cyber Security capstone project.

## 📁 Project Structure

```
icu_digital_twin/
├── main.py                  # App entry point + router registration
├── dependencies.py          # JWT auth, fake DB, shared helpers
├── requirements.txt
└── routers/
    ├── auth.py              # Login, logout, get me, update role
    ├── patients.py          # Admit, list, get, discharge
    ├── vitals.py            # Push vitals, history, critical, WebSocket
    ├── resources.py         # ICU resources + what-if simulation
    ├── ai.py                # Risk score, length of stay, AI alerts
    ├── siem.py              # Security events, anomalies, audit log, incidents
    └── chatbot.py           # NLP chatbot interface
```

## 🚀 Run the API

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open: http://localhost:8000/docs

## 🔐 Test Credentials

| Username   | Password      | Role       |
|------------|---------------|------------|
| dr.ahmad   | password123   | clinician  |
| admin      | admin123      | admin      |

## 📡 Key Endpoints

### Auth
- `POST /auth/login` — Get JWT token
- `GET  /auth/me`    — Get current user

### ICU Patients
- `POST   /icu/patients`          — Admit patient
- `GET    /icu/patients`          — List all patients
- `GET    /icu/patients/{id}`     — Patient + latest vitals
- `DELETE /icu/patients/{id}`     — Discharge patient

### Vitals
- `POST /icu/vitals/{id}`          — Push new reading
- `GET  /icu/vitals/{id}/history`  — Historical vitals
- `GET  /icu/vitals/critical`      — All critical patients
- `WS   /icu/vitals/ws/{id}`       — Live stream

### AI Models
- `GET /icu/ai/risk/{id}`          — Sepsis + deterioration score
- `GET /icu/ai/predict/los/{id}`   — Length of stay prediction
- `GET /icu/ai/alerts`             — All active AI alerts

### Resources
- `GET  /icu/resources`            — Full resource overview
- `PUT  /icu/resources/{id}`       — Update resource status
- `POST /icu/simulation/whatif`    — What-if scenario

### SIEM Security
- `POST /siem/events`              — Log security event
- `GET  /siem/anomalies`           — Isolation Forest anomaly detection
- `GET  /siem/alerts`              — Active security alerts
- `PUT  /siem/alerts/{id}/acknowledge`
- `GET  /siem/audit-log`           — Full compliance audit trail
- `POST /siem/incidents`           — Create incident report

### Chatbot
- `POST   /chatbot/query`          — Natural language question
- `DELETE /chatbot/history`        — Clear session

## 🤖 AI Models

| Endpoint | Simulated As | Production Model |
|----------|-------------|-----------------|
| `/icu/ai/risk` | Rule-based scoring | XGBoost / LSTM |
| `/icu/ai/predict/los` | Heuristic | Gradient Boosting |
| `/siem/anomalies` | Threshold rules | Isolation Forest |
| `/chatbot/query` | Keyword matching | LangChain + LLM |
