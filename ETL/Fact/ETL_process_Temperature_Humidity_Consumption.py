# Required library imports
import os
import pandas as pd
import logging
import traceback
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

import sys
# Add parent directory to path to make ETL a proper package import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ETL.utils.logging_utils import log_error, setup_logging, send_error_summary
from ETL.db.base import get_session, init_db
from ETL.db.models import FactEnergyConsumption
from ETL.Dim.DimDate import get_or_create_date
from ETL.Dim.DimTime import get_or_create_time
from config import ensure_installed, Config
from ETL.db.models import DimDate, DimTime

# Ensure required packages are installed
ensure_installed('pandas')
ensure_installed('sqlalchemy')

# Define subfolder structure
SUBFOLDERS = {
    "BellevueConso": ["Consumption", "Temperature", "Humidity"]
}

def get_files_by_category():
    """Get all CSV files organized by category"""
    files_by_category = {}
    
    if not Config.CLEAN_DATA_DIR or not os.path.exists(Config.CLEAN_DATA_DIR):
        log_error("File Search", f"Directory does not exist: {Config.CLEAN_DATA_DIR}")
        return files_by_category
        
    # List all folders in the cleaned_data directory to find the most recent one
    data_folders = [f for f in os.listdir(Config.BASE_DIR) if f.startswith('cleaned_data_')]
    if not data_folders:
        log_error("File Search", "No cleaned_data folders found")
        return files_by_category
        
    # Get the most recent data folder
    latest_data_folder = sorted(data_folders)[-1]
    actual_data_dir = os.path.join(Config.BASE_DIR, latest_data_folder)
    
    logging.info(f"Using data folder: {actual_data_dir}")
    
    # Process BellevueConso folder
    main_folder_path = os.path.join(actual_data_dir, "BellevueConso")
    if not os.path.exists(main_folder_path):
        log_error("File Search", f"Main folder path does not exist: {main_folder_path}")
        return files_by_category
        
    files_by_category["BellevueConso"] = {}
    
    # Get all CSV files and categorize them by type
    for f in os.listdir(main_folder_path):
        if f.endswith('.csv'):
            for category in SUBFOLDERS["BellevueConso"]:
                if category.lower() in f.lower():
                    if category not in files_by_category["BellevueConso"]:
                        files_by_category["BellevueConso"][category] = []
                    files_by_category["BellevueConso"][category].append(os.path.join(main_folder_path, f))
                    logging.info(f"Found {category} file: {f}")
    
    return files_by_category

def get_value_for_date_time(df, date_str, time_str, column_name):
    """Get value from dataframe for specific date and time"""
    if df is None:
        return None
    
    matches = df[(df['Date'] == date_str) & (df['Time'] == time_str)]
    return matches[column_name].iloc[0] if not matches.empty and column_name in matches.columns else None

def validate_energy_row(consumption_val, temperature_val, humidity_val):
    """Validate a single row of energy data"""
    try:
        # Check if all required values exist
        if any(val is None or pd.isna(val) for val in [consumption_val, temperature_val, humidity_val]):
            return False, "Missing required values"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_energy_files(session, consumption_df, temp_df, humidity_df):
    """Process energy consumption files and insert into database using ORM"""
    try:
        stats = {'processed': 0, 'inserted': 0, 'duplicates': 0, 'invalid': 0}
        
        # Preload all dimension data to reduce database queries
        logging.info("Preloading dimension data...")
        
        # Fetch all dates from dimension table
        all_dates = {}
        for date in session.query(DimDate).all():
            date_str = f"{date.year:04d}-{date.month:02d}-{date.day:02d}"
            all_dates[date_str] = date.id_date
        logging.info(f"Preloaded {len(all_dates)} dates")
        
        # Fetch all times from dimension table
        all_times = {}
        for time in session.query(DimTime).all():
            time_str = f"{time.hour:02d}:{time.minute:02d}:00"
            all_times[time_str] = time.id_time
        logging.info(f"Preloaded {len(all_times)} times")
        
        # Create a set for duplicate checking
        logging.info("Preloading existing records for duplicate checking...")
        existing_records = set()
        for record in session.query(
            FactEnergyConsumption.id_date,
            FactEnergyConsumption.id_time
        ).all():
            existing_records.add((record.id_date, record.id_time))
        
        logging.info(f"Preloaded {len(existing_records)} existing records for duplicate checking")
        
        # Batch insertion
        batch_size = 1000
        records_to_insert = []
        
        # Process each row in consumption dataframe
        for _, row in consumption_df.iterrows():
            stats['processed'] += 1
            
            try:
                date_str = row['Date']
                time_str = row['Time']
                
                # Get dimension IDs
                if date_str not in all_dates:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    date_id = get_or_create_date(session, date_obj)
                    all_dates[date_str] = date_id
                date_id = all_dates[date_str]
                
                if time_str not in all_times:
                    time_obj = datetime.strptime(time_str, '%H:%M:%S')
                    time_id = get_or_create_time(session, time_obj)
                    all_times[time_str] = time_id
                time_id = all_times[time_str]
                
                # Check for duplicates in memory
                record_key = (date_id, time_id)
                if record_key in existing_records:
                    stats['duplicates'] += 1
                    continue
                
                # Get related temperature and humidity values
                temperature = get_value_for_date_time(temp_df, date_str, time_str, 'Value')
                humidity = get_value_for_date_time(humidity_df, date_str, time_str, 'Value')
                
                # Validate data
                is_valid, message = validate_energy_row(row['Value'], temperature, humidity)
                if not is_valid:
                    stats['invalid'] += 1
                    log_error("Data Validation", f"Invalid row: {message} - Date: {date_str}, Time: {time_str}")
                    continue
                
                # Add to batch for insertion
                records_to_insert.append({
                    'id_date': date_id,
                    'id_time': time_id,
                    'energy_consumed': float(row['Value']), # Use updated energy_consumed
                    'temperature': float(temperature),
                    'humidity': float(humidity)
                })
                stats['inserted'] += 1
                
                # Add to existing records to prevent future duplicates
                existing_records.add(record_key)
                
                # Batch commit
                if len(records_to_insert) >= batch_size:
                    try:
                        session.bulk_insert_mappings(FactEnergyConsumption, records_to_insert)
                        session.commit()
                        logging.info(f"Committed batch of {len(records_to_insert)} records (total processed: {stats['processed']})")
                        records_to_insert = []
                    except Exception as batch_error:
                        error_trace = traceback.format_exc()
                        log_error("Batch Commit", f"Error during batch commit: {batch_error}", error_trace)
                        session.rollback()
                        records_to_insert = []
                
            except Exception as row_error:
                error_trace = traceback.format_exc()
                log_error("Row Processing", f"Error processing row: {row_error}", error_trace)
                continue
        
        # Commit any remaining rows in the final batch
        if records_to_insert:
            try:
                session.bulk_insert_mappings(FactEnergyConsumption, records_to_insert)
                session.commit()
                logging.info(f"Committed final batch of {len(records_to_insert)} records")
            except Exception as batch_error:
                error_trace = traceback.format_exc()
                log_error("Final Batch Commit", f"Error during final batch commit: {batch_error}", error_trace)
                session.rollback()
        
        # Log statistics
        logging.info("=== Summary ===")
        logging.info(f"Total Processed: {stats['processed']}")
        logging.info(f"Total Inserted: {stats['inserted']}")
        logging.info(f"Total Duplicates: {stats['duplicates']}")
        logging.info(f"Total Invalid: {stats['invalid']}")
        logging.info("===============")
        
        return stats
    
    except SQLAlchemyError as e:
        error_trace = traceback.format_exc()
        log_error("Database", f"Database Error: {str(e)}", error_trace)
        session.rollback()
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Processing", f"Processing Error: {str(e)}", error_trace)
        session.rollback()
        raise

def populate_dim_tables_and_facts():
    """Main ETL process to populate all tables"""
    # Setup logging for the ETL process
    setup_logging("Energy Consumption ETL")
    logging.info("Starting Energy Consumption ETL process")
    
    # Check for files first before attempting any database operations
    organized_files = get_files_by_category()
    
    # If the directory is empty, log an error and exit
    if not organized_files or not any(organized_files.values()):
        log_error("File Search", "No files found in the directory")
        send_error_summary("Energy Consumption ETL")
        logging.info("Energy Consumption ETL process completed without processing any files")
        return
    
    # Now that we know files exist, proceed with database initialization
    try:
        init_db()
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Database Init", f"Error initializing database: {str(e)}", error_trace)
        send_error_summary("Energy Consumption ETL")
        return
    
    session = get_session()
    if not session:
        log_error("Database Session", "Failed to create database session")
        send_error_summary("Energy Consumption ETL")
        return
    
    try:
        total_stats = {'processed': 0, 'inserted': 0, 'duplicates': 0, 'invalid': 0}
        
        for main_folder, categories in organized_files.items():
            # Process files by date to ensure we have all related data
            consumption_files = categories.get('Consumption', [])
            
            for consumption_file in consumption_files:
                try:
                    file_date = os.path.basename(consumption_file).split('-')[0]
                    
                    # Find corresponding temperature and humidity files
                    temp_file = next((f for f in categories.get('Temperature', []) 
                                    if file_date in f), None)
                    humidity_file = next((f for f in categories.get('Humidity', []) 
                                        if file_date in f), None)
                    
                    if not all([temp_file, humidity_file]):
                        log_error("File Matching", f"Missing temperature or humidity data for {file_date}")
                        continue
                    
                    logging.info(f"Processing files for date {file_date}")
                    logging.info(f"Consumption: {os.path.basename(consumption_file)}")
                    logging.info(f"Temperature: {os.path.basename(temp_file)}")
                    logging.info(f"Humidity: {os.path.basename(humidity_file)}")
                    
                    # Read and preprocess all files
                    consumption_df = pd.read_csv(consumption_file)
                    temp_df = pd.read_csv(temp_file)
                    humidity_df = pd.read_csv(humidity_file)
                    
                    # Process the files
                    file_stats = process_energy_files(session, consumption_df, temp_df, humidity_df)
                    
                    # Update total stats
                    for key in total_stats:
                        if key in file_stats:
                            total_stats[key] += file_stats[key]
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    log_error("File Processing", f"Error processing file {consumption_file}: {str(e)}", error_trace)
                    continue
        
        # Only show summary if files were processed
        if total_stats['processed'] > 0:
            logging.info("=== Final Summary ===")
            logging.info(f"Total Processed: {total_stats['processed']}")
            logging.info(f"Total Inserted: {total_stats['inserted']}")
            logging.info(f"Total Duplicates: {total_stats['duplicates']}")
            logging.info(f"Total Invalid: {total_stats['invalid']}")
            logging.info("===================")
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("ETL Process", f"ETL Error: {str(e)}", error_trace)
        if session:
            session.rollback()
    finally:
        if session:
            session.close()
        
        # Send error summary if there were any errors
        send_error_summary("Energy Consumption ETL")
        
        # Log completion message
        logging.info("Energy Consumption ETL process completed")

if __name__ == "__main__":
    populate_dim_tables_and_facts()
