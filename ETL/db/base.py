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
    
    # Create tables if they don't exist
    Base.metadata.create_all(engine)
    print("Database tables created or verified.")
    
    # check if all the tables are created
    tables = Base.metadata.tables.keys()
    if not tables:
        print("No tables found in the database.")
    else:
        print(f"Tables found: {', '.join(tables)}")
        
    return engine
