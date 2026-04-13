import httpx
import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, patients, vitals, resources, ai, siem, chatbot
from routers.realtime_router import router as realtime_router, _kafka_inference_listener
from pipeline import start_pipeline, stop_pipeline  
from tee_enclave import get_enclave                      
from routers.tee_router import router as tee_router 
from dependencies import fake_patients_db, fake_resources_db

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
    enc = get_enclave()                             
    enc.seal({                                    
        "patients":        dict(fake_patients_db),  
        "resources":       dict(fake_resources_db), 
        "simulation_log":  [],                     
    }) 
    yield
    await stop_pipeline()   

app = FastAPI(
    title="ICU Digital Twin API",
    description="Hospital ICU Digital Twin Simulation System with SIEM Security",
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
app.include_router(siem.router,      prefix="/siem",     tags=["SIEM Security"])
app.include_router(chatbot.router,   prefix="/chatbot",  tags=["Chatbot"])
app.include_router(realtime_router)
app.include_router(tee_router)

@app.get("/", tags=["Root"])
def root():
    return {"message": "ICU Digital Twin API is running "}
