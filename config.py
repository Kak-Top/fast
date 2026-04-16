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
    CACHE_TTL_SECONDS: int = 300  # 5 min
    
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