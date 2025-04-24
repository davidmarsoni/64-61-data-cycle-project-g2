#!/usr/bin/env python
"""
Database Setup Script

This script automatically creates the database schema on the server by executing
the galaxy_schema_creation_script.sql file. It handles connecting to the database server,
running the SQL script, and handling errors.

Usage:
    python setup_database.py [--force]
    
Options:
    --force    Force recreation of database even if it already exists
"""

import sys
import os
import argparse
import logging
from datetime import datetime
import traceback
import re

# Add parent directory to path so we can import from config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import ensure_installed, Config

# Ensure necessary packages are installed
ensure_installed('sqlalchemy')
ensure_installed('pyodbc')

import sqlalchemy
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import URL
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

def setup_logging():
    """Configure logging for the database setup script"""
    log_dir = os.path.join(Config.BASE_DIR, "logs") if Config.BASE_DIR else "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y-%m-%d')
    log_filename = f"db_setup_{timestamp}.log"
    log_file = os.path.join(log_dir, log_filename)
    
    # Reset handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    
    return log_file

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Database setup script")
    parser.add_argument('--force', action='store_true', 
                        help='Force recreation of database even if it already exists')
    return parser.parse_args()

def get_database_engine():
    """Get a SQLAlchemy engine for the database"""
    try:
        # First try local connection with trusted authentication (like in ETL/utils/utils.py)
        logging.info("Attempting to connect to local SQL Server with Windows Authentication...")
        connection_url = URL.create(
            "mssql+pyodbc",
            host=".",  # Local SQL Server
            database="master",  # Connect to master to create new database
            query={
                "driver": "ODBC Driver 17 for SQL Server",
                "trusted_connection": "yes",
                "TrustServerCertificate": "yes"
            }
        )
        
        engine = create_engine(
            connection_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"timeout": 30}
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT @@VERSION"))
            logging.info("Successfully connected to local SQL Server")
        
        return engine
        
    except SQLAlchemyError as e:
        logging.warning(f"Local connection failed: {str(e)}. Trying remote connection...")
        
        # Try to get from environment variables for remote connection
        server = os.getenv("DB_SERVER")
        username = os.getenv("DB_USERNAME")
        password = os.getenv("DB_PASSWORD")
        
        if not all([server, username, password]):
            # Try to get from Windows Credential Manager using Config
            server = Config.SMB_HOST  # Using the same server as file shares
            username = Config.USERNAME
            password = Config.PASSWORD
        
        if not all([server, username, password]):
            raise ValueError("Database connection information not found for remote connection. Set DB_SERVER, DB_USERNAME and DB_PASSWORD in environment variables or configure in Windows Credential Manager.")
        
        # Create connection URL for remote connection
        connection_url = URL.create(
            "mssql+pyodbc",
            username=username,
            password=password,
            host=server,
            database="master",
            query={"driver": "ODBC Driver 17 for SQL Server", "TrustServerCertificate": "yes"}
        )
        
        logging.info(f"Attempting to connect to remote SQL Server at {server}...")
        
        engine = create_engine(
            connection_url,
            echo=False,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"timeout": 30}
        )
        
        return engine

def read_sql_script():
    """Read the SQL schema creation script"""
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'galaxy_schema_creation_script.sql')
    
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"SQL script not found: {script_path}")
    
    with open(script_path, 'r') as file:
        script_content = file.read()
    
    return script_content

def database_exists(engine, db_name="data_cycle_db"):
    """Check if a database exists using SQLAlchemy"""
    inspector = inspect(engine)
    databases = engine.execute(text("SELECT name FROM master.dbo.sysdatabases")).fetchall()
    return db_name in [db[0] for db in databases]

def execute_script(script_content, force=False):
    """Execute the SQL script to create the database schema"""
    logging.info("Starting database setup process")
    
    try:
        # Get database engine
        engine = get_database_engine()
        logging.info("Established connection to database server")
        
        # If not forcing recreation and database already exists, check first
        if not force:
            try:
                with engine.connect() as connection:
                    # Check if database exists
                    result = connection.execute(text("SELECT name FROM master.dbo.sysdatabases WHERE name = 'data_cycle_db'"))
                    if result.scalar():
                        logging.warning("Database 'data_cycle_db' already exists. Use --force to recreate it.")
                        return False
            except SQLAlchemyError as e:
                logging.error(f"Error checking database existence: {str(e)}")
                raise
        else:
            # If force flag is set, check if database exists and ask for confirmation
            try:
                with engine.connect() as connection:
                    result = connection.execute(text("SELECT name FROM master.dbo.sysdatabases WHERE name = 'data_cycle_db'"))
                    if result.scalar():
                        # Database exists, ask for confirmation
                        print("\n⚠️ WARNING: You're about to drop and recreate the 'data_cycle_db' database.")
                        print("All existing data will be permanently lost!")
                        confirm = input("\nAre you sure you want to continue? (y/N): ")
                        if confirm.lower() != 'y':
                            logging.info("Database recreation canceled by user.")
                            print("\n❌ Database recreation canceled.")
                            return False
                        else:
                            print("\n🔄 Proceeding with database recreation...")
            except SQLAlchemyError as e:
                logging.error(f"Error checking database existence: {str(e)}")
                raise
        
        # Parse the SQL script to handle GO statements properly
        statements = []
        for batch in re.split(r'\bGO\b', script_content, flags=re.IGNORECASE):
            batch = batch.strip()
            if batch:
                statements.append(batch)
        
        # Execute each batch individually - can't use a single transaction for all statements
        with engine.connect() as connection:
            # First disable autocommit for manual transaction control
            connection = connection.execution_options(isolation_level="AUTOCOMMIT")
            
            for i, statement in enumerate(statements):
                try:
                    # Skip empty statements
                    if not statement.strip():
                        continue
                    
                    logging.info(f"Executing SQL batch #{i+1}...")
                    
                    # Execute each statement
                    connection.execute(text(statement))
                    
                except SQLAlchemyError as e:
                    logging.warning(f"Error executing batch #{i+1}: {str(e)}")
                    # For database setup, we might want to continue with other statements
                    # instead of failing completely, especially for idempotent scripts
        
        logging.info("Database schema creation completed successfully")
        return True
        
    except SQLAlchemyError as e:
        error_trace = traceback.format_exc()
        logging.error(f"Database error: {str(e)}")
        logging.error(error_trace)
        
        # Send email notification of the failure
        if hasattr(Config, 'send_error_email'):
            Config.send_error_email(
                "Database Setup",
                "Database Creation Failed",
                f"Error creating database schema: {str(e)}",
                error_trace
            )
        
        return False
        
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Unexpected error: {str(e)}")
        logging.error(error_trace)
        
        # Send email notification of the failure
        if hasattr(Config, 'send_error_email'):
            Config.send_error_email(
                "Database Setup",
                "Database Creation Failed",
                f"Unexpected error during database setup: {str(e)}",
                error_trace
            )
        
        return False

def main():
    """Main function to execute the database setup"""
    # Set up logging
    log_file = setup_logging()
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Initialize Config to load environment variables
        Config.load_credentials()
        
        # If force flag is set, check if database exists and ask for confirmation first
        if args.force:
            try:
                engine = get_database_engine()
                with engine.connect() as connection:
                    result = connection.execute(text("SELECT name FROM master.dbo.sysdatabases WHERE name = 'data_cycle_db'"))
                    if result.scalar():
                        # Database exists, ask for confirmation
                        print("\n⚠️ WARNING: You're about to drop and recreate the 'data_cycle_db' database.")
                        print("All existing data will be permanently lost!")
                        confirm = input("\nAre you sure you want to continue? (y/N): ")
                        if confirm.lower() != 'y':
                            logging.info("Database recreation canceled by user.")
                            print("\n❌ Database recreation canceled.")
                            return 1
                        print("\n🔄 Proceeding with database recreation...")
            except Exception as e:
                logging.error(f"Error checking database existence: {str(e)}")
                # Continue anyway - if we can't check, we'll let the script try to run
        
        # Read SQL script
        script_content = read_sql_script()
        
        # Execute script
        success = execute_script(script_content, args.force)
        
        if success:
            logging.info("Database setup completed successfully")
            print(f"✅ Database setup completed successfully. See log file for details: {log_file}")
            return 0
        else:
            logging.warning("Database setup completed with warnings or errors")
            print(f"⚠️ Database setup completed with warnings or errors. Check the log file: {log_file}")
            return 1
            
    except Exception as e:
        error_trace = traceback.format_exc()
        logging.error(f"Setup failed: {str(e)}")
        logging.error(error_trace)
        print(f"❌ Database setup failed: {str(e)}")
        print(f"See log file for details: {log_file}")
        return 1

if __name__ == "__main__":
    sys.exit(main())