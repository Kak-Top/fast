# config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str = "NEOcare TurboQuant"
    API_V1_STR: str = "/api/v1"
    
    # TurboQuant
    POLAR_INPUT_DIM: int = 8  # HR, SpO2, Temp, Resp, BP_sys, BP_dia, Weight, GestAge
    QUANT_BITS: int = 3       # Ultra-compression
    CKKS_POLY_DEGREE: int = 8192
    CKKS_SCALE: int = 2**40
    
    # Performance
    CACHE_TTL_SECONDS: int = 300  # 5 min (TurboQuant encoded-vitals cache)
    AI_CONCURRENCY_LIMIT: int = 3  # Max parallel AI/CKKS computations
    RISK_CACHE_TTL_SECONDS: int = 5  # In-memory risk result cache (seconds)
    TICK_INTERVAL: int = 15  # Simulator vitals generation interval (seconds)
    PATIENT_POLL_INTERVAL: int = 60  # Patient discovery polling interval (seconds)
    
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_SERVERS", "localhost:9092")
    KAFKA_VITALS_TOPIC: str = "neonatal.vitals.raw"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key")
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()
