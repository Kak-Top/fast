# models.py - TOP OF FILE
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, LargeBinary, JSON, ForeignKey
from sqlalchemy.orm import relationship
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    full_name = Column(String)
    role = Column(String)
    hashed_password = Column(String)
    disabled = Column(Boolean, default=False)

class Patient(Base):
    __tablename__ = "patients"

    patient_id = Column(String, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    diagnosis = Column(String)
    bed_id = Column(String)
    admitted_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String)

    # Relationships
    vitals = relationship("Vital", back_populates="patient", cascade="all, delete-orphan")
    resources = relationship("Resource", back_populates="patient")

class Vital(Base):
    __tablename__ = "vitals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("patients.patient_id"))
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    heart_rate = Column(Float)
    blood_pressure_sys = Column(Float)
    blood_pressure_dia = Column(Float)
    spo2 = Column(Float)
    respiratory_rate = Column(Float)
    temperature = Column(Float)

    patient = relationship("Patient", back_populates="vitals")

class Resource(Base):
    __tablename__ = "resources"

    resource_id = Column(String, primary_key=True, index=True)
    type = Column(String)  # bed, ventilator, monitor
# models.py — ADD THIS NEW TABLE

class TrainedModel(Base):
    """Stores trained custom model metadata + pickle blob."""
    __tablename__ = "trained_models"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String(100), unique=True, index=True, nullable=False)
    model_name = Column(String(50), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    
    # Performance metrics
    accuracy = Column(Float, nullable=False, default=0.0)
    f1_score = Column(Float, default=0.0)
    auc_roc = Column(Float, default=0.0)
    n_samples = Column(Integer, default=0)
    n_features = Column(Integer, default=8)
    
    # Feature names as JSON array
    feature_names = Column(JSON, default=list)
    
    # Security/TurboQuant metadata
    turboquant_enabled = Column(Boolean, default=False)
    ckks_enabled = Column(Boolean, default=False)
    compression_ratio = Column(String(20), default="1.0x")
    vram_saved_percent = Column(Float, default=0.0)
    encoding_latency_ms = Column(Float, default=0.0)
    
    # Who trained it and when
    description = Column(Text, default="")
    trained_at = Column(Float, default=0.0)  # Unix timestamp
    trained_by = Column(String(100), default="")
    is_active = Column(Boolean, default=False)  # Only ONE should be active
    
    # The actual model — stored as pickle bytes
    estimator_pickle = Column(LargeBinary, nullable=True)
    
       # NEW CODE (WORKS IN PYTHON 3.11)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    status = Column(String)  # occupied, available, in_use
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=True)

    patient = relationship("Patient", back_populates="resources")
