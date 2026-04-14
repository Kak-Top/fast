import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Neon DB URL from the environment
# IMPORTANT: Neon provides URLs that start with "postgres://". 
# For asyncpg, it MUST start with "postgresql+asyncpg://". We handle this below.
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")

if SQLALCHEMY_DATABASE_URL:
    if SQLALCHEMY_DATABASE_URL.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
    elif SQLALCHEMY_DATABASE_URL.startswith("postgresql://"):
        SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    
    # asyncpg doesn't support the raw sslmode query parameter
    SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("?sslmode=require", "")


# Initialize the async engine
# Neon requires SSL, so we use connect_args to enforce it safely
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URL or "postgresql+asyncpg://user:password@localhost/dbname",
    echo=True, # Set to False in production
    connect_args={"ssl": "require"} if "neon.tech" in (SQLALCHEMY_DATABASE_URL or "") else {}
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Declarative base for our models
Base = declarative_base()

# Dependency to yield database sessions via FastAPI
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
