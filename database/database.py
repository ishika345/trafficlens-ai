from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database.models import Base
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/trafficlens"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency injected into FastAPI route handlers."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Call once on startup to create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)