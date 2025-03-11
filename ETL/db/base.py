from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from ETL.utils.utils import get_connection_string

Base = declarative_base()

def get_engine():
    """Create SQLAlchemy engine with connection string from utils"""
    connection_string = get_connection_string()
    return create_engine(connection_string)

def get_session():
    """Create a new Session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def init_db():
    """Initialize database tables if they don't exist"""
    engine = get_engine()
    # Import all models to ensure they're registered with Base
    from ETL.db.models import DimDate, DimTime, DimSite, FactMeteoSwissData
    
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    print("Database tables created or verified.")
    
    return engine
