"""
Shared in-memory fake database and utility functions.
Replace these with real DB (PostgreSQL + SQLAlchemy) in production.
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

# ── JWT Config ──────────────────────────────────────────────────────────────
SECRET_KEY = "icu-digital-twin-secret-key-change-in-prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ── In-Memory Fake DB ────────────────────────────────────────────────────────
fake_users_db = {
    "dr.ahmad": {
        "user_id": "u001",
        "username": "dr.ahmad",
        "full_name": "Dr. Ahmad Khalil",
        "role": "clinician",
        "hashed_password": pwd_context.hash("password123"),
        "disabled": False,
    },
    "admin": {
        "user_id": "u002",
        "username": "admin",
        "full_name": "IT Admin",
        "role": "admin",
        "hashed_password": pwd_context.hash("admin123"),
        "disabled": False,
    },
     "dr.sara": {
        "user_id": "u003",
        "username": "dr.sara",
        "full_name": "Dr. Sara Mohammed",
        "role": "clinician",
        "hashed_password": pwd_context.hash("sara123"),
        "disabled": False,
    },
}

fake_patients_db = {
    "P001": {
        "patient_id": "P001",
        "name": "Khalid Al-Mansouri",
        "age": 67,
        "gender": "Male",
        "diagnosis": "Respiratory Failure",
        "bed_id": "ICU-01",
        "admitted_at": "2025-02-28T08:00:00",
        "status": "critical",
    },
    "P002": {
        "patient_id": "P002",
        "name": "Layla Hassan",
        "age": 54,
        "gender": "Female",
        "diagnosis": "Septic Shock",
        "bed_id": "ICU-02",
        "admitted_at": "2025-03-01T14:30:00",
        "status": "stable",
    },
}

fake_vitals_db = {
    "P001": [
        {"timestamp": "2025-03-02T08:00:00", "heart_rate": 112, "blood_pressure_sys": 88,
         "blood_pressure_dia": 55, "spo2": 91, "respiratory_rate": 26, "temperature": 38.9},
        {"timestamp": "2025-03-02T08:15:00", "heart_rate": 118, "blood_pressure_sys": 85,
         "blood_pressure_dia": 52, "spo2": 89, "respiratory_rate": 28, "temperature": 39.1},
    ],
    "P002": [
        {"timestamp": "2025-03-02T08:00:00", "heart_rate": 88, "blood_pressure_sys": 110,
         "blood_pressure_dia": 70, "spo2": 97, "respiratory_rate": 18, "temperature": 37.2},
    ],
}

fake_resources_db = {
    "ICU-01": {"resource_id": "ICU-01", "type": "bed", "status": "occupied", "patient_id": "P001"},
    "ICU-02": {"resource_id": "ICU-02", "type": "bed", "status": "occupied", "patient_id": "P002"},
    "ICU-03": {"resource_id": "ICU-03", "type": "bed", "status": "available", "patient_id": None},
    "VENT-01": {"resource_id": "VENT-01", "type": "ventilator", "status": "in_use", "patient_id": "P001"},
    "VENT-02": {"resource_id": "VENT-02", "type": "ventilator", "status": "available", "patient_id": None},
    "MON-01":  {"resource_id": "MON-01",  "type": "monitor",    "status": "in_use", "patient_id": "P001"},
    "MON-02":  {"resource_id": "MON-02",  "type": "monitor",    "status": "in_use", "patient_id": "P002"},
}

fake_siem_events_db = []
fake_siem_incidents_db = []
revoked_tokens = set()

# ── Pydantic Token Schemas ───────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# ── Helpers ──────────────────────────────────────────────────────────────────
def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

from database import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from models import User

async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if token in revoked_tokens:
        raise HTTPException(status_code=401, detail="Token has been revoked")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    query = await db.execute(select(User).where(User.username == username))
    user = query.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
        
    return {
        "user_id": user.user_id,
        "username": user.username,
        "full_name": user.full_name,
        "role": user.role,
        "hashed_password": user.hashed_password,
        "disabled": user.disabled
    }

def require_role(*roles):
    def checker(current_user=Depends(get_current_user)):
        if current_user["role"] not in roles:
            raise HTTPException(status_code=403, detail=f"Access denied. Required roles: {list(roles)}")
        return current_user
    return checker
