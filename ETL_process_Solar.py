"""
ETL script to populate the FactSolarProduction table and related dimensions
using SQLAlchemy ORM.
"""

import os
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey, DateTime, inspect
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import exists
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("solar_production_etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global configuration
BASE_DIR = os.getenv('BASE_DIR', 'C:\\DataCollection')  # Valeur par défaut si non définie
current_date = datetime.now().strftime('%Y-%m-%d')
CLEAN_DATA_DIR = os.path.join(BASE_DIR, f"cleaned_data_{current_date}")

# Database connection
SERVER = os.getenv('DB_SERVER', '.')
DATABASE = os.getenv('DB_NAME', 'data_cycle_db')
CONNECTION_STRING = f'mssql+pyodbc://{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'

# SQLAlchemy setup
Base = declarative_base()

# Define ORM models matching the database schema
class DimDate(Base):
    __tablename__ = 'DimDate'
    
    id_date = Column(Integer, primary_key=True, autoincrement=True)
    year = Column(Integer, nullable=False)
    month = Column(Integer, nullable=False)
    day = Column(Integer, nullable=False)

class DimTime(Base):
    __tablename__ = 'DimTime'
    
    id_time = Column(Integer, primary_key=True, autoincrement=True)
    hour = Column(Integer, nullable=False)
    minute = Column(Integer, nullable=False)

class DimInverter(Base):
    __tablename__ = 'DimInverter'
    
    id_inverter = Column(Integer, primary_key=True, autoincrement=True)
    inverterName = Column(String(255), nullable=False)

class DimStatus(Base):
    __tablename__ = 'DimStatus'
    
    id_status = Column(Integer, primary_key=True, autoincrement=True)
    statusName = Column(String(255), nullable=False)

class FactSolarProduction(Base):
    __tablename__ = 'FactSolarProduction'
    
    id_FactSolarProduction = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_status = Column(Integer, ForeignKey('DimStatus.id_status'), nullable=True)
    id_inverter = Column(Integer, ForeignKey('DimInverter.id_inverter'), nullable=False)
    total_energy_produced = Column(Float, nullable=False)
    energy_produced = Column(Float, nullable=False)
    error_count = Column(Integer, default=0)


class SolarProductionETL:
    """ETL process for solar production data"""
    
    def __init__(self):
        """Initialize the ETL process"""
        self.engine = create_engine(CONNECTION_STRING)
        
        # Check if tables exist and create them if they don't
        self.create_tables_if_not_exist()
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Caches for dimension lookups
        self.date_cache = {}
        self.time_cache = {}
        self.inverter_cache = {}
        self.status_cache = {}
        
        # Load status descriptions
        self.status_descriptions = {
            0: "Standby",
            6: "Running", 
            14: "Error"
        }
        
        # Pre-load dimension caches
        self._preload_dimensions()
    
    def create_tables_if_not_exist(self):
        """Check if required tables exist and create them if they don't"""
        inspector = inspect(self.engine)
        tables = inspector.get_table_names()
        
        # Create all tables defined in Base if they don't exist
        if not all(table in tables for table in ['DimDate', 'DimTime', 'DimInverter', 'DimStatus', 'FactSolarProduction']):
            logger.info("Creating database tables that don't exist...")
            Base.metadata.create_all(self.engine)
            logger.info("Tables created successfully")
        else:
            logger.info("All required tables already exist")
            
            # Check if columns match our model
            for table_name in ['FactSolarProduction', 'DimDate', 'DimTime', 'DimInverter', 'DimStatus']:
                if table_name in tables:
                    columns = [col['name'] for col in inspector.get_columns(table_name)]
                    logger.info(f"Table {table_name} columns: {', '.join(columns)}")
                    
                    # Verify key columns for FactSolarProduction
                    if table_name == 'FactSolarProduction':
                        expected_columns = ['id_FactSolarProduction', 'id_date', 'id_time', 'id_inverter', 'id_status', 'total_energy_produced', 'energy_produced', 'error_count']
                        missing = [col for col in expected_columns if col not in columns]
                        if missing:
                            logger.warning(f"Missing columns in {table_name}: {', '.join(missing)}")
                            raise ValueError(f"Database schema mismatch. Missing columns in {table_name}: {', '.join(missing)}")

    def _preload_dimensions(self):
        """Pre-load dimension tables to minimize database queries"""
        logger.info("Pre-loading dimension data...")
        
        # Load dates
        for date_row in self.session.query(DimDate).all():
            date_key = f"{date_row.year}-{date_row.month:02d}-{date_row.day:02d}"
            self.date_cache[date_key] = date_row.id_date
            
        # Load times
        for time_row in self.session.query(DimTime).all():
            time_key = f"{time_row.hour:02d}:{time_row.minute:02d}:00"
            self.time_cache[time_key] = time_row.id_time
            
        # Load inverters
        for inverter_row in self.session.query(DimInverter).all():
            self.inverter_cache[inverter_row.inverterName] = inverter_row.id_inverter
            
        # Load statuses
        for status_row in self.session.query(DimStatus).all():
            for status_code, status_name in self.status_descriptions.items():
                if status_row.statusName == status_name:
                    self.status_cache[status_code] = status_row.id_status
        
        logger.info(f"Loaded {len(self.date_cache)} dates, {len(self.time_cache)} times, "
                   f"{len(self.inverter_cache)} inverters, and {len(self.status_cache)} statuses")

    def get_or_create_date_id(self, date_str):
        """Get or create date dimension entry"""
        if date_str in self.date_cache:
            return self.date_cache[date_str]
            
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Check if date exists in database
        date_entry = self.session.query(DimDate).filter_by(
            year=date_obj.year, 
            month=date_obj.month, 
            day=date_obj.day
        ).first()
        
        if date_entry:
            self.date_cache[date_str] = date_entry.id_date
            return date_entry.id_date
            
        # Create new date
        new_date = DimDate(
            year=date_obj.year,
            month=date_obj.month,
            day=date_obj.day
        )
        self.session.add(new_date)
        self.session.flush()  # Get the generated ID
        
        self.date_cache[date_str] = new_date.id_date
        return new_date.id_date
        
    def get_or_create_time_id(self, time_str):
        """Get or create time dimension entry"""
        if time_str in self.time_cache:
            return self.time_cache[time_str]
            
        time_obj = datetime.strptime(time_str, '%H:%M:%S')
        
        # Check if time exists
        time_entry = self.session.query(DimTime).filter_by(
            hour=time_obj.hour,
            minute=time_obj.minute
        ).first()
        
        if time_entry:
            self.time_cache[time_str] = time_entry.id_time
            return time_entry.id_time
            
        # Create new time
        new_time = DimTime(
            hour=time_obj.hour,
            minute=time_obj.minute
        )
        self.session.add(new_time)
        self.session.flush()
        
        self.time_cache[time_str] = new_time.id_time
        return new_time.id_time
        
    def get_or_create_inverter_id(self, inverter_id):
        """Get or create inverter dimension entry"""
        inverter_name = f"INV-{inverter_id}"
        
        if inverter_name in self.inverter_cache:
            return self.inverter_cache[inverter_name]
            
        # Check if inverter exists
        inverter_entry = self.session.query(DimInverter).filter_by(
            inverterName=inverter_name
        ).first()
        
        if inverter_entry:
            self.inverter_cache[inverter_name] = inverter_entry.id_inverter
            return inverter_entry.id_inverter
            
        # Create new inverter
        new_inverter = DimInverter(
            inverterName=inverter_name
        )
        self.session.add(new_inverter)
        self.session.flush()
        
        self.inverter_cache[inverter_name] = new_inverter.id_inverter
        return new_inverter.id_inverter
        
    def get_or_create_status_id(self, status_code):
        """Get or create status dimension entry"""
        if status_code in self.status_cache:
            return self.status_cache[status_code]
            
        status_name = self.status_descriptions.get(status_code, f"Unknown-{status_code}")
        
        # Check if status exists
        status_entry = self.session.query(DimStatus).filter_by(
            statusName=status_name
        ).first()
        
        if status_entry:
            self.status_cache[status_code] = status_entry.id_status
            return status_entry.id_status
            
        # Create new status
        new_status = DimStatus(
            statusName=status_name
        )
        self.session.add(new_status)
        self.session.flush()
        
        self.status_cache[status_code] = new_status.id_status
        return new_status.id_status
        
    def record_exists(self, date_id, time_id, inverter_id):
        """Check if a record already exists to avoid duplicates"""
        # Utiliser count() pour SQL Server au lieu de exists()
        return self.session.query(FactSolarProduction).filter(
            FactSolarProduction.id_date == date_id,
            FactSolarProduction.id_time == time_id,
            FactSolarProduction.id_inverter == inverter_id 
        ).count() > 0
        
    def process_solar_panels_data(self):
        """Process data from the SOLAR PANELS files"""
        try:
            # Path to Solarlogs folder
            solar_logs_dir = os.path.join(CLEAN_DATA_DIR, "Solarlogs")
            if not os.path.exists(solar_logs_dir):
                logger.warning(f"Solarlogs directory not found: {solar_logs_dir}")
                return
                
            # Find all solar panel files (min*.csv pattern)
            solar_files = []
            for root, _, files in os.walk(solar_logs_dir):
                for file in files:
                    if file.startswith("min") and file.endswith('.csv'):
                        solar_files.append(os.path.join(root, file))
            
            if not solar_files:
                logger.warning("No SOLAR PANELS files found!")
                return
                
            batch_count = 0
            total_records = 0
            batch_size = 500  # Commit every 500 records
            
            for file_path in solar_files:
                logger.info(f"Processing {file_path}")
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check for required columns
                    required_columns = ['Date', 'Time', 'INV', 'Pac']
                    if not all(col in df.columns for col in required_columns):
                        missing = [col for col in required_columns if col not in df.columns]
                        logger.warning(f"Missing required columns in {file_path}: {', '.join(missing)}")
                        continue
                    
                    # Handle 'DaySum' column if missing
                    if 'DaySum' not in df.columns:
                        logger.info(f"'DaySum' column not found in {file_path}, using Pac as total_energy_produced")
                        df['DaySum'] = df['Pac']
                    
                    # Standardize date and time formats with explicit format to avoid warnings
                    try:
                        # Handle different possible date formats
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        
                        # Try with different time formats to handle various input formats
                        time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
                        for time_format in time_formats:
                            try:
                                df['Time'] = pd.to_datetime(df['Time'], format=time_format, errors='coerce').dt.strftime('%H:%M:%S')
                                # If successful (no NaT values), break out of the loop
                                if not df['Time'].isna().any():
                                    break
                            except ValueError:
                                continue
                        
                        # If still has NaT values, try the flexible parser as last resort
                        if df['Time'].isna().any():
                            df['Time'] = pd.to_datetime(df['Time'], errors='coerce').dt.strftime('%H:%M:%S')
                            
                        # Drop rows where date or time couldn't be parsed
                        df = df.dropna(subset=['Date', 'Time'])
                    except Exception as e:
                        logger.warning(f"Error parsing dates/times in {file_path}: {str(e)}")
                    
                    # Process each row
                    for _, row in df.iterrows():
                        # Skip rows with missing critical data
                        if pd.isna(row['Date']) or pd.isna(row['Time']) or pd.isna(row['INV']) or pd.isna(row['Pac']):
                            continue
                            
                        date_id = self.get_or_create_date_id(row['Date'])
                        time_id = self.get_or_create_time_id(row['Time'])
                        inverter_id = self.get_or_create_inverter_id(row['INV'])
                        
                        # Skip if record already exists
                        if self.record_exists(date_id, time_id, inverter_id):
                            continue
                            
                        # Get status ID (if available)
                        status_id = None
                        if pd.notna(row.get('Status')):
                            status_id = self.get_or_create_status_id(int(row['Status']))
                            
                        # Create new fact record
                        new_fact = FactSolarProduction(
                            id_date=date_id,
                            id_time=time_id,
                            id_inverter=inverter_id,
                            total_energy_produced=float(row['DaySum']),
                            energy_produced=float(row['Pac']),
                            id_status=status_id,
                            error_count=int(row.get('Error', 0)) if pd.notna(row.get('Error')) else 0
                        )
                        self.session.add(new_fact)
                        
                        batch_count += 1
                        total_records += 1
                        
                        # Commit in batches
                        if batch_count >= batch_size:
                            self.session.commit()
                            logger.info(f"Committed batch of {batch_count} records, total: {total_records}")
                            batch_count = 0
                            
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    self.session.rollback()
                    continue
                    
            # Commit any remaining records
            if batch_count > 0:
                self.session.commit()
                logger.info(f"Committed final batch of {batch_count} records, total: {total_records}")
                
            logger.info(f"Completed processing SOLAR PANELS data. Total records: {total_records}")
            
        except Exception as e:
            logger.error(f"Error in process_solar_panels_data: {str(e)}")
            self.session.rollback()
            raise
            
    def process_solar_history_data(self):
        """Process data from the SOLARPANELS HISTORY files"""
        try:
            # Path to Solarlogs folder
            solar_logs_dir = os.path.join(CLEAN_DATA_DIR, "Solarlogs")
            if not os.path.exists(solar_logs_dir):
                logger.warning(f"Solarlogs directory not found: {solar_logs_dir}")
                return
                
            # Find all solar panel history files (*.csv files with dates like DD.MM.YYYY-PV.csv)
            history_files = []
            for root, _, files in os.walk(solar_logs_dir):
                for file in files:
                    if "-PV.csv" in file and any(c == '.' for c in file[:10]):  # Match date pattern and -PV.csv
                        history_files.append(os.path.join(root, file))
            
            if not history_files:
                logger.warning("No SOLARPANELS HISTORY files found!")
                return
                
            batch_count = 0
            total_records = 0
            batch_size = 500  # Commit every 500 records
            
            for file_path in history_files:
                logger.info(f"Processing history file {file_path}")
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Check for required columns
                    required_columns = ['Date', 'Heure', 'INV', 'Variation']
                    if not all(col in df.columns for col in required_columns):
                        missing = [col for col in required_columns if col not in df.columns]
                        logger.warning(f"Missing required columns in {file_path}: {', '.join(missing)}")
                        continue
                    
                    # Standardize date and time formats with explicit format to avoid warnings
                    try:
                        # Handle different possible date formats
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
                        
                        # For 'Heure' column (time column in French)
                        time_formats = ['%H:%M:%S', '%H:%M', '%I:%M:%S %p', '%I:%M %p']
                        for time_format in time_formats:
                            try:
                                df['Time'] = pd.to_datetime(df['Heure'], format=time_format, errors='coerce').dt.strftime('%H:%M:%S')
                                # If successful (no NaT values), break out of the loop
                                if not df['Time'].isna().any():
                                    break
                            except ValueError:
                                continue
                        
                        # If still has NaT values, try the flexible parser as last resort
                        if 'Time' not in df.columns or df['Time'].isna().any():
                            df['Time'] = pd.to_datetime(df['Heure'], errors='coerce').dt.strftime('%H:%M:%S')
                            
                        # Drop rows where date or time couldn't be parsed
                        df = df.dropna(subset=['Date', 'Time'])
                    except Exception as e:
                        logger.warning(f"Error parsing dates/times in {file_path}: {str(e)}")
                    
                    # Filter for power measurements
                    if 'UnitDisplay' in df.columns:
                        power_data = df[df['UnitDisplay'].str.contains('kW|watt|kWh', case=False, na=False)]
                    else:
                        power_data = df  # Use all data if no UnitDisplay column
                    
                    # Process each row
                    for _, row in power_data.iterrows():
                        # Skip rows with missing critical data
                        if pd.isna(row['Date']) or pd.isna(row['Time']) or pd.isna(row['INV']) or pd.isna(row['Variation']):
                            continue
                            
                        date_id = self.get_or_create_date_id(row['Date'])
                        time_id = self.get_or_create_time_id(row['Time'])
                        inverter_id = self.get_or_create_inverter_id(row['INV'])
                        
                        # Skip if record already exists (might have been added from SOLAR PANELS)
                        if self.record_exists(date_id, time_id, inverter_id):
                            continue
                        
                        # Use Variation for both energy values since historical data might not have separate total
                        variation_value = float(row['Variation'])
                            
                        # Create new fact record with power variation data
                        new_fact = FactSolarProduction(
                            id_date=date_id,
                            id_time=time_id,
                            id_inverter=inverter_id,
                            energy_produced=variation_value,
                            total_energy_produced=variation_value,  # Use same value for both since historical data
                            id_status=None,
                            error_count=0
                        )
                        self.session.add(new_fact)
                        
                        batch_count += 1
                        total_records += 1
                        
                        # Commit in batches
                        if batch_count >= batch_size:
                            self.session.commit()
                            logger.info(f"Committed batch of {batch_count} records, total: {total_records}")
                            batch_count = 0
                            
                except Exception as e:
                    logger.error(f"Error processing history file {file_path}: {str(e)}")
                    self.session.rollback()
                    continue
                    
            # Commit any remaining records
            if batch_count > 0:
                self.session.commit()
                logger.info(f"Committed final batch of {batch_count} records, total: {total_records}")
                
            logger.info(f"Completed processing SOLARPANELS HISTORY data. Total records: {total_records}")
            
        except Exception as e:
            logger.error(f"Error in process_solar_history_data: {str(e)}")
            self.session.rollback()
            raise
            
    def run(self):
        """Run the complete ETL process"""
        try:
            logger.info("Starting Solar Production ETL process")
            
            # Process primary data source first
            self.process_solar_panels_data()
            
            # Process secondary/historical data
            self.process_solar_history_data()
            
            logger.info("Solar Production ETL process completed successfully")
            
        except Exception as e:
            logger.error(f"ETL process failed: {str(e)}")
            self.session.rollback()
        finally:
            self.session.close()


if __name__ == "__main__":
    etl = SolarProductionETL()
    etl.run()