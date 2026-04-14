import asyncio
from database import engine, Base, AsyncSessionLocal
from models import User, Patient, Vital, Resource
from dependencies import fake_users_db, fake_patients_db, fake_vitals_db, fake_resources_db
import datetime

async def seed_database():
    print("Creating database tables...")
    async with engine.begin() as conn:
        # Warning: In a real app, use Alembic. For this transition, we drop and create.
        await conn.run_sync(Base.metadata.create_all)
    
    print("Tables created successfully!")
    print("Seeding data from fake dictionaries to PostgreSQL...")

    async with AsyncSessionLocal() as session:
        # Seed Users
        for username, data in fake_users_db.items():
            user = User(
                user_id=data["user_id"],
                username=data["username"],
                full_name=data["full_name"],
                role=data["role"],
                hashed_password=data["hashed_password"],
                disabled=data["disabled"]
            )
            session.add(user)
        
        # Seed Patients
        for patient_id, data in fake_patients_db.items():
            admitted_dt = datetime.datetime.fromisoformat(data["admitted_at"])
            patient = Patient(
                patient_id=data["patient_id"],
                name=data["name"],
                age=data["age"],
                gender=data["gender"],
                diagnosis=data["diagnosis"],
                bed_id=data["bed_id"],
                admitted_at=admitted_dt,
                status=data["status"]
            )
            session.add(patient)

        # Seed Vitals
        for patient_id, vitals_list in fake_vitals_db.items():
            for v_data in vitals_list:
                vital_dt = datetime.datetime.fromisoformat(v_data["timestamp"])
                vital = Vital(
                    patient_id=patient_id,
                    timestamp=vital_dt,
                    heart_rate=v_data["heart_rate"],
                    blood_pressure_sys=v_data["blood_pressure_sys"],
                    blood_pressure_dia=v_data["blood_pressure_dia"],
                    spo2=v_data["spo2"],
                    respiratory_rate=v_data["respiratory_rate"],
                    temperature=v_data["temperature"]
                )
                session.add(vital)

        # Seed Resources
        for resource_id, data in fake_resources_db.items():
            resource = Resource(
                resource_id=data["resource_id"],
                type=data["type"],
                status=data["status"],
                patient_id=data["patient_id"]
            )
            session.add(resource)

        try:
            await session.commit()
            print("Successfully migrated all fake data to Neon PostgreSQL!")
        except Exception as e:
            await session.rollback()
            print(f"Error seeding data (maybe it already exists?): {e}")

if __name__ == "__main__":
    asyncio.run(seed_database())
