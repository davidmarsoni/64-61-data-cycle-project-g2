# Required library imports
import os
import pandas as pd
import logging
import traceback
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from ETL.utils.logging_utils import log_error, setup_logging, send_error_summary
from ETL.db.base import get_session, init_db
from ETL.db.models import FactMeteoSwissData
from ETL.Dim.DimDate import get_or_create_date
from ETL.Dim.DimTime import get_or_create_time
from ETL.Dim.DimSite import get_or_create_site
from config import ensure_installed, Config
from ETL.db.models import DimDate, DimTime, DimSite

# Ensure required packages are installed
ensure_installed('pandas')
ensure_installed('sqlalchemy')

SUBFOLDERS = {
    "Meteo": ["Pred"]
}

def get_files_by_category():
    """Get all CSV files organized by category"""
    files_by_category = {}
    
    if not Config.CLEAN_DATA_DIR or not os.path.exists(Config.CLEAN_DATA_DIR):
        log_error("File Search", f"Directory does not exist: {Config.CLEAN_DATA_DIR}")
        return files_by_category
        
    data_folders = [f for f in os.listdir(Config.BASE_DIR) if f.startswith('cleaned_data_')]
    if not data_folders:
        log_error("File Search", "No cleaned_data folders found")
        return files_by_category
        
    latest_data_folder = sorted(data_folders)[-1]
    actual_data_dir = os.path.join(Config.BASE_DIR, latest_data_folder)
    
    logging.info(f"Using data folder: {actual_data_dir}")
    
    # Process Meteo folder
    meteo_folder_path = os.path.join(actual_data_dir, "Meteo")
    if not os.path.exists(meteo_folder_path):
        log_error("File Search", f"Meteo folder path does not exist: {meteo_folder_path}")
        return files_by_category
        
    files_by_category["Meteo"] = {}
    
    # Get all Pred CSV files
    for f in os.listdir(meteo_folder_path):
        if f.endswith('.csv') and f.startswith('Pred_'):
            if "Pred" not in files_by_category["Meteo"]:
                files_by_category["Meteo"]["Pred"] = []
            files_by_category["Meteo"]["Pred"].append(os.path.join(meteo_folder_path, f))
            logging.info(f"Found meteo prediction file: {f}")
    
    return files_by_category

def get_value_for_date_time(df, date_str, time_str, column_name):
    """Get value from dataframe for specific date and time"""
    if df is None:
        return None
    
    matches = df[(df['Date'] == date_str) & (df['Time'] == time_str)]
    return matches[column_name].iloc[0] if not matches.empty and column_name in matches.columns else None

def validate_meteo_row(row):
    """Validate a single row of meteorological data"""
    try:
        # Check if all required columns exist
        required_columns = ['Date', 'Time', 'Site', 'Prediction', 
                          'PRED_T_2M_ctrl', 'PRED_RELHUM_2M_ctrl', 
                          'PRED_TOT_PREC_ctrl', 'PRED_GLOB_ctrl']
        if not all(col in row.index for col in required_columns):
            return False, "Missing required columns"
        
        # Validate prediction number
        if not isinstance(row['Prediction'], (int, float)) or pd.isna(row['Prediction']):
            return False, "Invalid prediction number"
        
        # Validate measurements
        measurements = {
            'temperature': row['PRED_T_2M_ctrl'],
            'humidity': row['PRED_RELHUM_2M_ctrl'],
            'rain': row['PRED_TOT_PREC_ctrl'],
            'radiation': row['PRED_GLOB_ctrl']
        }
        
        if any(pd.isna(value) for value in measurements.values()):
            return False, "Missing measurement values"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_meteo_files(session, meteo_folder):
    """Process meteorological data files and insert into database using ORM"""
    try:
        # Get all prediction files
        pred_files = [f for f in os.listdir(meteo_folder) if f.startswith('Pred_') and f.endswith('.csv')]
        stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'unchanged': 0}
        
        logging.info(f"Found {len(pred_files)} files to process")
        
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
        
        # Fetch all sites from dimension table
        all_sites = {}
        for site in session.query(DimSite).all():
            all_sites[site.siteName] = site.id_site
        logging.info(f"Preloaded {len(all_sites)} sites")
        
        # Create a dictionary for existing records to enable updates
        logging.info("Preloading existing records for updating...")
        existing_records = {}
        for record in session.query(
            FactMeteoSwissData
        ).all():
            record_key = (record.id_date, record.id_time, record.id_site, record.numPrediction)
            existing_records[record_key] = record
        
        logging.info(f"Preloaded {len(existing_records)} existing records for checking")
        
        for i, pred_file in enumerate(pred_files, 1):
            logging.info(f"Processing file {i}/{len(pred_files)}: {pred_file}")
            file_path = os.path.join(meteo_folder, pred_file)
            try:
                meteo_df = pd.read_csv(file_path)
                file_stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'unchanged': 0}
                
                # Standardize dates and times
                meteo_df['Date'] = pd.to_datetime(meteo_df['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
                meteo_df['Time'] = pd.to_datetime(meteo_df['Time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
                
                # Create dimension mappings
                mappings = {
                    'date': {},
                    'time': {},
                    'site': {}
                }
                
                # Batch insertion
                batch_size = 1000
                
                # Process each row
                for _, row in meteo_df.iterrows():
                    file_stats['processed'] += 1
                    
                    try:
                        # Get dimension IDs
                        date_str = row['Date']
                        if date_str not in mappings['date']:
                            if date_str in all_dates:
                                mappings['date'][date_str] = all_dates[date_str]
                            else:
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                date_id = get_or_create_date(session, date_obj)
                                mappings['date'][date_str] = date_id
                                all_dates[date_str] = date_id
                        date_id = mappings['date'][date_str]
                        
                        time_str = row['Time']
                        if time_str not in mappings['time']:
                            if time_str in all_times:
                                mappings['time'][time_str] = all_times[time_str]
                            else:
                                time_obj = datetime.strptime(time_str, '%H:%M:%S')
                                time_id = get_or_create_time(session, time_obj)
                                mappings['time'][time_str] = time_id
                                all_times[time_str] = time_id
                        time_id = mappings['time'][time_str]
                        
                        site_name = row['Site']
                        if site_name not in mappings['site']:
                            if site_name in all_sites:
                                mappings['site'][site_name] = all_sites[site_name]
                            else:
                                site_id = get_or_create_site(session, site_name)
                                mappings['site'][site_name] = site_id
                                all_sites[site_name] = site_id
                        site_id = mappings['site'][site_name]
                        
                        prediction_num = int(row['Prediction'])
                        
                        # New data values
                        new_temperature = float(row['PRED_T_2M_ctrl'])
                        new_humidity = float(row['PRED_RELHUM_2M_ctrl'])
                        new_rain = float(row['PRED_TOT_PREC_ctrl'])
                        new_radiation = float(row['PRED_GLOB_ctrl'])
                        
                        # Check if record exists and update it if values changed
                        record_key = (date_id, time_id, site_id, prediction_num)
                        
                        if record_key in existing_records:
                            # Get existing record
                            existing_record = existing_records[record_key]
                            
                            # Check if any values have changed
                            values_changed = (
                                abs(existing_record.temperature - new_temperature) > 0.001 or
                                abs(existing_record.humidity - new_humidity) > 0.001 or
                                abs(existing_record.rain - new_rain) > 0.001 or
                                abs(existing_record.radiation - new_radiation) > 0.001
                            )
                            
                            if values_changed:
                                # Update existing record with new values
                                existing_record.temperature = new_temperature
                                existing_record.humidity = new_humidity
                                existing_record.rain = new_rain
                                existing_record.radiation = new_radiation
                                file_stats['updated'] += 1
                            else:
                                file_stats['unchanged'] += 1
                        else:
                            # Add new record to batch for insertion
                            new_record = FactMeteoSwissData(
                                id_date=date_id,
                                id_time=time_id,
                                id_site=site_id,
                                numPrediction=prediction_num,
                                temperature=new_temperature,
                                humidity=new_humidity,
                                rain=new_rain,
                                radiation=new_radiation
                            )
                            session.add(new_record)
                            file_stats['inserted'] += 1
                            
                            # Add to existing records dictionary
                            existing_records[record_key] = new_record
                        
                    except Exception as row_error:
                        error_trace = traceback.format_exc()
                        log_error("Row Processing", f"Error processing row in file {pred_file}: {row_error}", error_trace)
                        continue
                    
                    # Commit every batch_size records
                    if (file_stats['processed'] % batch_size) == 0:
                        try:
                            session.commit()
                            logging.info(f"Committed batch at {file_stats['processed']} records")
                        except Exception as batch_error:
                            error_trace = traceback.format_exc()
                            log_error("Batch Commit", f"Error during batch commit: {batch_error}", error_trace)
                            session.rollback()
                
                # Commit any remaining changes if there are not enough for a full batch
                try:
                    session.commit()
                    logging.info(f"Committed final batch")
                except Exception as batch_error:
                    error_trace = traceback.format_exc()
                    log_error("Final Batch Commit", f"Error during final batch commit: {batch_error}", error_trace)
                    session.rollback()
                
                # Update file statistics
                logging.info(f"File complete: {pred_file}")
                logging.info(f"Rows processed: {file_stats['processed']}")
                logging.info(f"Rows inserted: {file_stats['inserted']}")
                logging.info(f"Rows updated: {file_stats['updated']}")
                logging.info(f"Rows unchanged: {file_stats['unchanged']}")
                
                # Update total statistics
                stats['processed'] += file_stats['processed']
                stats['inserted'] += file_stats['inserted']
                stats['updated'] += file_stats['updated']
                stats['unchanged'] += file_stats['unchanged']
                
            except Exception as file_error:
                error_trace = traceback.format_exc()
                log_error("File Processing", f"Error processing file {pred_file}: {file_error}", error_trace)
                continue
            
        logging.info("=== Final Summary ===")
        logging.info(f"Total Processed: {stats['processed']}")
        logging.info(f"Total Inserted: {stats['inserted']}")
        logging.info(f"Total Updated: {stats['updated']}")
        logging.info(f"Total Unchanged: {stats['unchanged']}")
        logging.info("===================")
        
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
    """Main ETL process"""
    # Setup logging for the ETL process
    setup_logging("Meteo ETL")
    logging.info("Starting Meteo ETL process")
    
    # First ensure tables exist
    try:
        init_db()
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Database Init", f"Error initializing database: {str(e)}", error_trace)
        send_error_summary("Meteo ETL")
        return
        
    session = get_session()
    if not session:
        log_error("Database Session", "Failed to create database session")
        send_error_summary("Meteo ETL")
        return
    
    try:
        meteo_folder = os.path.join(Config.BASE_DIR, Config.CLEAN_DATA_DIR, "Meteo")
        
        if os.path.exists(meteo_folder):
            process_meteo_files(session, meteo_folder)
        else:
            log_error("Folder Access", f"Error: No Meteo folder in {Config.CLEAN_DATA_DIR}")
            
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("ETL Process", f"ETL Error: {str(e)}", error_trace)
        session.rollback()
    finally:
        session.close()
        
        # Send error summary if there were any errors
        send_error_summary("Meteo ETL")
        
        logging.info("Meteo ETL process completed")

if __name__ == "__main__":
    populate_dim_tables_and_facts()
