from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import auth, patients, vitals, resources, ai, siem, chatbot

app = FastAPI(
    title="ICU Digital Twin API",
    description="Hospital ICU Digital Twin Simulation System with SIEM Security",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/auth",     tags=["Auth"])
app.include_router(patients.router,  prefix="/icu",      tags=["ICU Patients"])
app.include_router(vitals.router,    prefix="/icu",      tags=["Vitals"])
app.include_router(resources.router, prefix="/icu",      tags=["ICU Resources"])
app.include_router(ai.router,        prefix="/icu/ai",   tags=["AI Models"])
app.include_router(siem.router,      prefix="/siem",     tags=["SIEM Security"])
app.include_router(chatbot.router,   prefix="/chatbot",  tags=["Chatbot"])

@app.get("/", tags=["Root"])
def root():
    return {"message": "ICU Digital Twin API is running 🏥"}
