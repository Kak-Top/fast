import httpx
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, patients, vitals, resources, ai, chatbot
from routers.realtime_router import router as realtime_router, _kafka_inference_listener
from pipeline import start_pipeline, stop_pipeline  
from dependencies import fake_patients_db, fake_resources_db
from routers.tee import router as tee_router
from middleware.tee_gateway import TEEGatewayMiddleware
from routers.custom_model import router as custom_model_router

async def keep_alive():
    await asyncio.sleep(30)
    while True:
        try:
            url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
            async with httpx.AsyncClient() as client:
                await client.get(f"{url}/")
                print("Keep-alive ping sent ✓")
        except Exception as e:
            print(f"Keep-alive failed: {e}")
        await asyncio.sleep(5 * 60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(keep_alive())
    asyncio.create_task(_kafka_inference_listener())
    await start_pipeline() 

    # ── TEE: Seal initial state at boot ──────────────
    try:
        from services.merkle_audit import get_merkle_tree
        from utils.proof import seal_data

        # Seal the initial DB state with HMAC proof
        initial_state = {
            "patients":        dict(fake_patients_db),  
            "resources":       dict(fake_resources_db), 
            "simulation_log":  [],                     
        }
        sealed = seal_data(initial_state)
        print(f"✓ Initial state sealed with HMAC proof: {sealed['proof'][:16]}...")

        # Log boot event to Merkle audit trail
        merkle = get_merkle_tree()
        merkle.add_entry(
            event_type="SYSTEM_BOOT",
            actor="system",
            data={
                "action": "initial_state_sealed",
                "patient_count": len(fake_patients_db),
                "resource_count": len(fake_resources_db),
            },
        )
        print(f"✓ Boot event logged to Merkle tree (root: {merkle.root_hash[:16]}...)")
    except Exception as e:
        print(f"⚠ TEE boot sealing failed: {e}")

    yield

    await stop_pipeline()   


app = FastAPI(
    title="ICU Digital Twin API",
    description="Hospital ICU Digital Twin Simulation System with TEE Security",
    version="1.0.0",
    lifespan=lifespan
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
app.include_router(custom_model_router, prefix="/icu/ai/models/custom", tags=["Custom AI Models"])
app.include_router(chatbot.router,   prefix="/chatbot",  tags=["Chatbot"])
app.include_router(realtime_router)

# ── TEE Routes ───────────────────────────────────────
app.include_router(tee_router)

# ── TEE Middleware (must be LAST add_middleware call) ─
app.add_middleware(TEEGatewayMiddleware)


@app.get("/", tags=["Root"])
def root():
    return {"message": "ICU Digital Twin API is running 🔐 TEE Protected"}
