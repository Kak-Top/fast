from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, ForeignKey
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
    status = Column(String)  # occupied, available, in_use
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=True)

    patient = relationship("Patient", back_populates="resources")
