# ICU Digital Twin — Secure FastAPI Backend

A production-style FastAPI backend for a Hospital ICU Digital Twin simulation platform, developed for a Cyber Security and Artificial Intelligence capstone project.

The system combines:

* Real-time ICU monitoring
* AI-powered clinical predictions
* Trusted Execution Environment (TEE) security
* Homomorphic encryption workflows
* Merkle-tree audit integrity
* Role-based access control
* WebSocket streaming
* Security anomaly detection

---

# Project Structure

```bash
icu_digital_twin/
├── main.py                     # FastAPI app entrypoint
├── dependencies.py             # JWT auth + shared dependencies
├── requirements.txt
│
├── routers/
│   ├── auth.py                 # Authentication & RBAC
│   ├── patients.py             # ICU patient management
│   ├── vitals.py               # Real-time vitals ingestion
│   ├── resources.py            # ICU resource management
│   ├── ai.py                   # AI risk prediction services
│   ├── custom_ai.py            # Train custom AI models
│   ├── oracle.py               # Multimodal AI Oracle engine
│   ├── labs.py                 # Clinical labs ingestion
│   ├── chatbot.py              # AI chatbot assistant
│   ├── tee.py                  # Trusted Execution Environment APIs
│   └── websocket.py            # Realtime WebSocket status
```

---

# Running the API

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Start the Server

```bash
uvicorn main:app --reload
```

Server:

```text
http://localhost:8000
```

Swagger Documentation:

```text
http://localhost:8000/docs
```

OpenAPI JSON:

```text
http://localhost:8000/openapi.json
```

---

# Authentication

The API uses:

* JWT Bearer Authentication
* OAuth2 Password Flow
* Role-Based Access Control (RBAC)

---

# Supported Roles

| Role        | Description                 |
| ----------- | --------------------------- |
| clinician   | ICU doctor / nurse access   |
| admin       | Full system administration  |
| manager     | ICU operational management  |
| it_security | Security & audit operations |

---

# Core Features

## ICU Digital Twin

* Patient admission & discharge
* Real-time vitals monitoring
* ICU bed/resource tracking
* Dashboard analytics
* Capacity planning simulations

## AI & Predictive Analytics

* Sepsis risk scoring
* Patient deterioration prediction
* ICU Length-of-Stay forecasting
* AI-generated alerts
* Custom AI model training
* Multimodal AI Oracle assessments

## Cyber Security & TEE

* HMAC-sealed responses
* Tamper-proof Merkle audit trail
* Attestation verification
* Homomorphic encryption support
* Secure encrypted inference
* Threat anomaly detection

## Realtime Infrastructure

* WebSocket vitals streaming
* Live Oracle task updates
* Session-aware chatbot
* Active WS connection monitoring

---

# API Endpoints

## Authentication

| Method | Endpoint                     | Description                   |
| ------ | ---------------------------- | ----------------------------- |
| POST   | `/auth/login`                | Login and receive JWT token   |
| POST   | `/auth/logout`               | Revoke current JWT token      |
| GET    | `/auth/me`                   | Current authenticated user    |
| PUT    | `/auth/users/{user_id}/role` | Update user role (admin only) |

---

## ICU Patients

| Method | Endpoint                     | Description                         |
| ------ | ---------------------------- | ----------------------------------- |
| GET    | `/icu/patients`              | List all ICU patients               |
| POST   | `/icu/patients`              | Admit new patient                   |
| GET    | `/icu/patients/{patient_id}` | Get patient details + latest vitals |
| DELETE | `/icu/patients/{patient_id}` | Discharge patient                   |

### Security Features

All patient endpoints:

* Generate HMAC integrity proofs
* Log operations into Merkle audit trail
* Support tamper verification

---

## Vitals Monitoring

| Method | Endpoint                           | Description         |
| ------ | ---------------------------------- | ------------------- |
| POST   | `/icu/vitals/{patient_id}`         | Push vitals reading |
| GET    | `/icu/vitals/{patient_id}/history` | Historical vitals   |
| GET    | `/icu/vitals/critical`             | Critical patients   |
| WS     | `/icu/vitals/ws/{patient_id}`      | Live vitals stream  |

---

## Clinical Labs

| Method | Endpoint                        | Description       |
| ------ | ------------------------------- | ----------------- |
| POST   | `/icu/labs/{patient_id}`        | Push lab results  |
| GET    | `/icu/labs/{patient_id}/latest` | Latest lab values |

### Supported Labs

* Glucose
* Creatinine
* WBC
* Lactate

---

## AI Models

| Method | Endpoint                           | Description                 |
| ------ | ---------------------------------- | --------------------------- |
| GET    | `/icu/ai/risk/{patient_id}`        | Sepsis & deterioration risk |
| GET    | `/icu/ai/predict/los/{patient_id}` | Predict ICU length of stay  |
| GET    | `/icu/ai/alerts`                   | Active AI alerts            |

### TurboQuant Support

Optional secure inference:

```http
?use_turboquant=true
```

Simulates secure low-bit encrypted inference acceleration.

---

## Custom AI Models

| Method | Endpoint                        | Description           |
| ------ | ------------------------------- | --------------------- |
| POST   | `/icu/ai/models/custom/train`   | Train custom AI model |
| GET    | `/icu/ai/models/custom/status`  | Training status       |
| POST   | `/icu/ai/models/custom/predict` | Run predictions       |
| DELETE | `/icu/ai/models/custom`         | Delete model          |

### Supported Features

* Synthetic dataset generation
* Live ICU data training
* TurboQuant optimization
* Configurable train/test split

---

## Oracle AI Engine

Multimodal AI assessment system combining:

* Medical imaging
* Clinical notes
* Patient vitals

| Method | Endpoint                        | Description             |
| ------ | ------------------------------- | ----------------------- |
| POST   | `/icu/ai/oracle/assess`         | Start Oracle assessment |
| GET    | `/icu/ai/oracle/task/{task_id}` | Poll Oracle task        |

### Oracle Workflow

1. Submit patient image + notes
2. Receive task ID
3. Subscribe via WebSocket or poll API
4. Receive AI assessment results

---

## ICU Resource Management

| Method | Endpoint                       | Description         |
| ------ | ------------------------------ | ------------------- |
| GET    | `/icu/resources`               | Resource overview   |
| POST   | `/icu/resources`               | Create ICU resource |
| GET    | `/icu/resources/{resource_id}` | Resource details    |
| PUT    | `/icu/resources/{resource_id}` | Update resource     |

### Resource Types

* Beds
* Ventilators
* Monitors

---

## ICU Simulation & Analytics

| Method | Endpoint                              | Description           |
| ------ | ------------------------------------- | --------------------- |
| POST   | `/icu/simulation/whatif`              | What-if simulation    |
| POST   | `/icu/simulation/capacity-planning`   | Predictive planning   |
| GET    | `/icu/dashboard/summary`              | ICU dashboard summary |
| GET    | `/icu/analytics/resource-utilization` | Utilization analytics |

### Example Scenarios

* Flu surge
* Equipment failure
* Staffing shortage
* Capacity expansion

---

## AI Chatbot

| Method | Endpoint           | Description        |
| ------ | ------------------ | ------------------ |
| POST   | `/chatbot/query`   | Query AI assistant |
| GET    | `/chatbot/history` | Session history    |
| DELETE | `/chatbot/history` | Clear history      |

### Chatbot Capabilities

* ICU data assistance
* Medical Q&A
* Code support
* General knowledge
* HTML-ready responses

---

## Trusted Execution Environment (TEE)

The platform includes a simulated Trusted Execution Environment layer for secure healthcare AI operations.

---

### Threat Detection

| Method | Endpoint      | Description              |
| ------ | ------------- | ------------------------ |
| POST   | `/tee/detect` | Threat anomaly detection |

---

### Encryption & Secure Inference

| Method | Endpoint                      | Description                     |
| ------ | ----------------------------- | ------------------------------- |
| POST   | `/tee/encrypt`                | Encrypt vitals                  |
| POST   | `/tee/decrypt`                | Decrypt prediction              |
| POST   | `/tee/encrypted_predict`      | AI prediction on encrypted data |
| POST   | `/tee/secure_vitals_pipeline` | Full secure pipeline            |
| GET    | `/tee/public_key`             | Retrieve HE public key          |

### Supported Encryption Modes

* AES-256
* CKKS Homomorphic Encryption

---

## Merkle Audit Trail

Immutable tamper-proof logging system.

| Method | Endpoint                      | Description            |
| ------ | ----------------------------- | ---------------------- |
| POST   | `/tee/audit/log`              | Append audit event     |
| GET    | `/tee/audit/root`             | Current Merkle root    |
| POST   | `/tee/audit/verify_integrity` | Verify tree integrity  |
| POST   | `/tee/audit/verify_proof`     | Verify inclusion proof |
| GET    | `/tee/audit/recent`           | Recent audit entries   |

---

## Remote Attestation

| Method | Endpoint             | Description                |
| ------ | -------------------- | -------------------------- |
| GET    | `/tee/attest`        | Generate attestation quote |
| POST   | `/tee/attest/verify` | Verify quote               |

---

## TEE Monitoring

| Method | Endpoint               | Description          |
| ------ | ---------------------- | -------------------- |
| GET    | `/tee/health`          | TEE health check     |
| GET    | `/tee/status`          | Detailed TEE status  |
| GET    | `/tee/security_report` | Full security report |

---

## Realtime APIs

| Method | Endpoint     | Description                  |
| ------ | ------------ | ---------------------------- |
| GET    | `/ws/status` | Active WebSocket connections |

---

# Security Architecture

## Implemented Security Controls

* JWT Authentication
* OAuth2 Authorization
* Role-Based Access Control
* HMAC Response Signing
* Merkle Tree Integrity Verification
* Simulated TEE Attestation
* Encrypted AI Inference
* Threat Detection Engine
* Audit Logging
* Secure WebSocket Sessions

---

# AI Architecture

| Feature          | Current Implementation   | Production Equivalent  |
| ---------------- | ------------------------ | ---------------------- |
| Risk Prediction  | Rule-Based + Heuristic   | XGBoost / LSTM         |
| LOS Prediction   | Heuristic Model          | Gradient Boosting      |
| Threat Detection | Threshold Analysis       | Isolation Forest       |
| Chatbot          | Keyword / Context Engine | LangChain + LLM        |
| Oracle           | Simulated Multimodal AI  | Vision-Language Model  |
| Secure Inference | CKKS Simulation          | Confidential GPU / SGX |

---

# Example Login Request

```bash
curl -X POST "http://localhost:8000/auth/login" \
-H "Content-Type: application/x-www-form-urlencoded" \
-d "username=admin&password=admin123"
```

---

# Example Authenticated Request

```bash
curl -X GET "http://localhost:8000/icu/patients" \
-H "Authorization: Bearer YOUR_TOKEN"
```

---

# Technologies Used

* FastAPI
* Python
* JWT / OAuth2
* WebSockets
* AI/ML Simulation
* Homomorphic Encryption Concepts
* Merkle Trees
* HMAC Integrity Verification
* ICU Digital Twin Simulation



* Production-grade security review
* HIPAA/GDPR compliance
