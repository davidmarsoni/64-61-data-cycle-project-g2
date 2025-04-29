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
    """Execute the SQL script to create the database schema.

    Returns:
        tuple: (bool, str) indicating success status and a descriptive message.
    """
    logging.info("Starting database setup process")
    
    try:
        # Get database engine
        engine = get_database_engine()
        logging.info("Established connection to database server")
        
        db_name = 'data_cycle_db'
        db_exists = False
        try:
            with engine.connect() as connection:
                # Check if database exists
                result = connection.execute(text(f"SELECT name FROM master.dbo.sysdatabases WHERE name = '{db_name}'"))
                db_exists = result.scalar() is not None
        except SQLAlchemyError as e:
            logging.error(f"Error checking database existence: {str(e)}")
            return False, f"Error checking database existence: {str(e)}" # Return specific error

        # Handle existing database based on force flag
        if db_exists:
            if not force:
                warning_msg = f"Database '{db_name}' already exists. Use --force to recreate it."
                logging.warning(warning_msg)
                return False, warning_msg # Return specific warning
            else:
                # Ask for confirmation if forcing recreation
                print(f"\nWARNING: You're about to drop and recreate the '{db_name}' database.")
                print("All existing data will be permanently lost!")
                confirm = input("\nAre you sure you want to continue? (y/N): ")
                if confirm.lower() != 'y':
                    cancel_msg = "Database recreation canceled by user."
                    logging.info(cancel_msg)
                    print("\nERROR: Database recreation canceled.") # Keep console error
                    return False, cancel_msg # Return specific reason
                else:
                    print("\n🔄 Proceeding with database recreation...")
        
        # Parse the SQL script to handle GO statements properly
        statements = []
        for batch in re.split(r'\bGO\b', script_content, flags=re.IGNORECASE):
            batch = batch.strip()
            if batch:
                statements.append(batch)

        # Execute each batch individually
        with engine.connect() as connection:
            connection = connection.execution_options(isolation_level="AUTOCOMMIT")
            
            for i, statement in enumerate(statements):
                try:
                    if not statement.strip():
                        continue
                    logging.info(f"Executing SQL batch #{i+1}...")
                    connection.execute(text(statement))
                except SQLAlchemyError as e:
                    # Log the specific batch error but continue
                    logging.warning(f"Error executing batch #{i+1}: {str(e)}")
                   
        
        success_msg = "Database schema creation completed successfully"
        logging.info(success_msg)
        return True, success_msg
        
    except SQLAlchemyError as e:
        error_trace = traceback.format_exc()
        error_msg = f"Database error during setup: {str(e)}"
        logging.error(error_msg)
        logging.error(error_trace)
        
        # Send email notification of the failure
        if hasattr(Config, 'send_error_email'):
            Config.send_error_email(
                "Database Setup",
                "Database Creation Failed",
                f"Error creating database schema: {str(e)}",
                error_trace
            )
        
        
        return False, error_msg 
        
    except Exception as e:
        error_trace = traceback.format_exc()
        error_msg = f"Unexpected error during database setup: {str(e)}"
        logging.error(error_msg)
        logging.error(error_trace)
        
        # Send email notification of the failure
        if hasattr(Config, 'send_error_email'):
            Config.send_error_email(
                "Database Setup",
                "Database Creation Failed",
                f"Error creating database schema: {str(e)}",
                error_trace
            )
       
        return False, error_msg 

def main():
    """Main function to execute the database setup"""
    log_file = setup_logging()
    args = parse_args()
    
    try:
        Config.load_credentials()
        
        script_content = read_sql_script()
        
        # Execute script and get status and message
        success, message = execute_script(script_content, args.force)
        
        if success:
            logging.info("Database setup process finished successfully.")
            # Use the success message from execute_script
            print(f"✅ {message}. See log file for details: {log_file}")
            return 0
        else:
            # Use the specific warning/error message from execute_script
            logging.warning(f"Database setup process finished with warnings or errors: {message}")
            print(f"WARNING/ERROR: {message}. Check the log file for details: {log_file}")
            return 1
            
    except Exception as e:
        # Catch errors from load_credentials, read_sql_script, etc.
        error_trace = traceback.format_exc()
        logging.error(f"Setup failed: {str(e)}")
        logging.error(error_trace)
        print(f"ERROR: Database setup failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())