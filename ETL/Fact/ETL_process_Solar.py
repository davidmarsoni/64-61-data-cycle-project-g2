# Required library imports
import os
import pandas as pd
import logging
import traceback
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from ETL.utils.logging_utils import log_error, setup_logging, send_error_summary
from ETL.db.base import get_session, init_db
from ETL.db.models import FactSolarProduction
from ETL.Dim.DimDate import get_or_create_date
from ETL.Dim.DimTime import get_or_create_time
from ETL.Dim.DimInverter import get_or_create_inverter
from ETL.Dim.DimStatus import get_or_create_status
from config import ensure_installed, Config
from ETL.db.models import DimDate, DimTime, DimInverter, DimStatus
  
# Ensure required packages are installed
ensure_installed('pandas')
ensure_installed('sqlalchemy')

SUBFOLDERS = {
    "Solarlogs": ["PV", "min"],
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
    
    # Process Solarlogs folder
    solarlogs_folder_path = os.path.join(actual_data_dir, "Solarlogs")
    if not os.path.exists(solarlogs_folder_path):
        log_error("File Search", f"Solarlogs folder path does not exist: {solarlogs_folder_path}")
        return files_by_category
        
    files_by_category["Solarlogs"] = {}
    
    # Initialize the categories
    for category in SUBFOLDERS["Solarlogs"]:
        files_by_category["Solarlogs"][category] = []
    
    # Look for files with category identifiers in their names in the Solarlogs folder
    for f in os.listdir(solarlogs_folder_path):
        if f.endswith('.csv'):
            file_path = os.path.join(solarlogs_folder_path, f)
            for category in SUBFOLDERS["Solarlogs"]:
                if category.lower() in f.lower():
                    files_by_category["Solarlogs"][category].append(file_path)
                    logging.info(f"Found {category} file: {f}")
    
    # Log summary of files found
    for category in SUBFOLDERS["Solarlogs"]:
        count = len(files_by_category["Solarlogs"][category])
        if count > 0:
            logging.info(f"Found {count} {category} files in Solarlogs folder")
        else:
            logging.warning(f"No {category} files found in Solarlogs folder")
    
    return files_by_category

def get_value_for_date_time(df, date_str, time_str, column_name):
    """Get value from dataframe for specific date and time"""
    if df is None:
        return None
    
    matches = df[(df['Date'] == date_str) & (df['Time'] == time_str)]
    return matches[column_name].iloc[0] if not matches.empty and column_name in matches.columns else None

def validate_solar_row(row, file_type="PV"):
    """Validate a single row of solar production data based on file type"""
    try:
        # Create a standardized row with all required fields
        standard_row = {}
        
        # Handle different file formats by mapping columns appropriately
        if file_type.lower() == "min":
            # Check if all required columns exist for min file format
            required_columns = ['Date', 'Time', 'INV', 'Pac', 'DaySum', 'Status', 'Error']
            if not all(col in row.index for col in required_columns):
                missing_cols = [col for col in required_columns if col not in row.index]
                return False, f"Missing required columns for min file: {missing_cols}"
                
            # Map min file columns to standard column names and handle missing values
            standard_row = {
                'Date': row['Date'],
                'Time': row['Time'],
                'InverterName': str(row['INV']),
                'StatusName': str(row['Status']),
                'TotalEnergyProduced': 0 if pd.isna(row['DaySum']) else row['DaySum'],
                'EnergyProduced': 0 if pd.isna(row['Pac']) else round((row['Pac']*5/60)/1000, 3),
                'ErrorCount': 0 if pd.isna(row['Error']) else row['Error']
            }
        else:  # PV file format
            # Check if all required columns exist for PV file format
            required_columns = ['Date', 'Time', 'Value']
            if not all(col in row.index for col in required_columns):
                missing_cols = [col for col in required_columns if col not in row.index]
                return False, f"Missing required columns for PV file: {missing_cols}"
                
            # Map PV file columns to standard column names and handle missing values
            standard_row = {
                'Date': row['Date'],
                'Time': row['Time'],
                'InverterName': 'History', # Default inverter name for the PV file
                'StatusName': '0',  # Default status for PV files
                'TotalEnergyProduced': 0,  # Default total energy produced for PV files
                'EnergyProduced': 0 if pd.isna(row['Value']) else row['Value'],
                'ErrorCount': 0  # Default error count for PV files
            }
        
        # Validate date format (should be a string that can be parsed)
        if isinstance(standard_row['Date'], str) and '.' in standard_row['Date']:
            # Handle European date format DD.MM.YY
            try:
                day, month, year = standard_row['Date'].split('.')
                if len(year) == 2:
                    year = f"20{year}"  # Assuming 20xx for two-digit years
                standard_row['Date'] = f"{year}-{month}-{day}"
            except Exception as e:
                return False, f"Invalid date format: {standard_row['Date']} - {str(e)}"
        
        # Validate measurements (convert to numeric if possible)
        try:
            # Convert string values to appropriate numeric types
            # If conversion fails, set to default value of 0
            try:
                standard_row['TotalEnergyProduced'] = float(standard_row['TotalEnergyProduced'])
            except (ValueError, TypeError):
                standard_row['TotalEnergyProduced'] = 0.0
                
            try:
                standard_row['EnergyProduced'] = float(standard_row['EnergyProduced'])
            except (ValueError, TypeError):
                standard_row['EnergyProduced'] = 0.0
                
            try:
                standard_row['ErrorCount'] = int(standard_row['ErrorCount'])
            except (ValueError, TypeError):
                standard_row['ErrorCount'] = 0
        except Exception as e:
            return False, f"Invalid measurement values, couldn't convert to numeric: {str(e)}"
        
        # Check for null values in key fields and set defaults
        if pd.isna(standard_row.get('InverterName')) or standard_row.get('InverterName') is None:
            if file_type.lower() == "pv":
                standard_row['InverterName'] = 'PV_Inverter'
            else:
                standard_row['InverterName'] = 'Unknown_Inverter'
            
        if pd.isna(standard_row.get('StatusName')) or standard_row.get('StatusName') is None:
            standard_row['StatusName'] = '0'
            
        if pd.isna(standard_row.get('TotalEnergyProduced')) or standard_row.get('TotalEnergyProduced') is None:
            standard_row['TotalEnergyProduced'] = 0.0
            
        if pd.isna(standard_row.get('EnergyProduced')) or standard_row.get('EnergyProduced') is None:
            standard_row['EnergyProduced'] = 0.0
            
        if pd.isna(standard_row.get('ErrorCount')) or standard_row.get('ErrorCount') is None:
            standard_row['ErrorCount'] = 0
        
        # Update the row with standardized values (to be used for DB insertion)
        for key, value in standard_row.items():
            row[key] = value
            
        return True, "Valid"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_solar_files(session, solarlogs_folder):
    """Process solar production data files and insert into database using ORM"""
    try:
        # Get all PV files from the PV subfolder
        pv_folder = os.path.join(solarlogs_folder, "PV")
        pv_files = []
        
        if os.path.exists(pv_folder):
            pv_files = [f for f in os.listdir(pv_folder) if f.endswith('.csv')]
            logging.info(f"Found {len(pv_files)} PV files to process")
        else:
            logging.warning(f"No PV subfolder found in {solarlogs_folder}. Will check for min files.")
        
        # Get all min files from the min subfolder
        min_folder = os.path.join(solarlogs_folder, "min")
        min_files = []
        
        if os.path.exists(min_folder):
            min_files = [f for f in os.listdir(min_folder) if f.endswith('.csv')]
            logging.info(f"Found {len(min_files)} min files to process")
        else:
            logging.warning(f"No min subfolder found in {solarlogs_folder}.")
            
        stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'unchanged': 0}
        
        # If no files at all, log and return early with a message
        if not pv_files and not min_files:
            logging.info("No PV or min files to process. ETL completed successfully.")
            return
            
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
        
        # Fetch all inverters from dimension table
        all_inverters = {}
        for inverter in session.query(DimInverter).all():
            all_inverters[inverter.inverterName] = inverter.id_inverter
        logging.info(f"Preloaded {len(all_inverters)} inverters")
        
        # Fetch all statuses from dimension table
        all_statuses = {}
        for status in session.query(DimStatus).all():
            all_statuses[status.statusName] = status.id_status
        logging.info(f"Preloaded {len(all_statuses)} statuses")
        
        # Create a dictionary for existing records to enable updates
        logging.info("Preloading existing records for updating...")
        existing_records = {}
        for record in session.query(
            FactSolarProduction
        ).all():
            record_key = (record.id_date, record.id_time, record.id_inverter, record.id_status)
            existing_records[record_key] = record
        
        logging.info(f"Preloaded {len(existing_records)} existing records for checking")
        
        # Process PV files if available
        if pv_files:
            logging.info("Processing PV files...")
            process_files_by_type(session, pv_folder, pv_files, "PV", all_dates, all_times, all_inverters, 
                                 all_statuses, existing_records, stats)
        
        # Process min files if available
        if min_files:
            logging.info("Processing min files...")
            process_files_by_type(session, min_folder, min_files, "min", all_dates, all_times, all_inverters, 
                                 all_statuses, existing_records, stats)
            
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

def process_files_by_type(session, folder_path, files, file_type, all_dates, all_times, all_inverters, 
                         all_statuses, existing_records, stats):
    """Process solar production files of a specific type (PV or min)"""
    file_stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'unchanged': 0}
    mappings = {
        'date': {},
        'time': {},
        'inverter': {},
        'status': {}
    }
    
    # Batch insertion
    batch_size = 1000
    
    for i, file_name in enumerate(files, 1):
        logging.info(f"Processing {file_type} file: {file_name}")
        file_path = os.path.join(folder_path, file_name)
        try:
            solar_df = pd.read_csv(file_path)
            
            # Standardize dates and times if they exist
            if 'Date' in solar_df.columns and len(solar_df) > 0:
                try:
                    # Try to detect the date format
                    if any('.' in str(date) for date in solar_df['Date'] if isinstance(date, str)):
                        # Handle DD.MM.YY format
                        solar_df['Date'] = pd.to_datetime(
                            solar_df['Date'], 
                            format='%d.%m.%y'
                        ).dt.strftime('%Y-%m-%d')
                    else:
                        # Standard YYYY-MM-DD format
                        solar_df['Date'] = pd.to_datetime(solar_df['Date']).dt.strftime('%Y-%m-%d')
                except Exception as e:
                    logging.warning(f"Error converting dates: {e}")
                    
            if 'Time' in solar_df.columns and len(solar_df) > 0:
                try:
                    solar_df['Time'] = pd.to_datetime(
                        solar_df['Time'], 
                        format='%H:%M:%S' if ':' in str(solar_df['Time'].iloc[0]) else None
                    ).dt.strftime('%H:%M:%S')
                except Exception as e:
                    logging.warning(f"Error converting times: {e}")
            
            # Process each row
            for idx, row in solar_df.iterrows():
                file_stats['processed'] += 1
                
                try:
                    # Validate the row first - this will also standardize column names
                    is_valid, message = validate_solar_row(row, file_type)
                    if not is_valid:
                        log_error("Data Validation", f"Invalid row in {file_name} (row {idx}): {message}")
                        continue
                    
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
                    
                    inverter_name = row['InverterName']
                    if inverter_name not in mappings['inverter']:
                        if inverter_name in all_inverters:
                            mappings['inverter'][inverter_name] = all_inverters[inverter_name]
                        else:
                            inverter_id = get_or_create_inverter(session, inverter_name)
                            mappings['inverter'][inverter_name] = inverter_id
                            all_inverters[inverter_name] = inverter_id
                    inverter_id = mappings['inverter'][inverter_name]
                    
                    status_name = row['StatusName']
                    if status_name not in mappings['status']:
                        if status_name in all_statuses:
                            mappings['status'][status_name] = all_statuses[status_name]
                        else:
                            status_id = get_or_create_status(session, status_name)
                            mappings['status'][status_name] = status_id
                            all_statuses[status_name] = status_id
                    status_id = mappings['status'][status_name]
                    
                    # New data values (already converted to correct types in the validation function)
                    new_totalEnergyProduced = row['TotalEnergyProduced']
                    new_energy_produced = row['EnergyProduced']
                    new_error_count = row['ErrorCount']
                    
                    # Check if record exists and update it if values changed
                    record_key = (date_id, time_id, inverter_id, status_id)
                    
                    if record_key in existing_records:
                        # Get existing record
                        existing_record = existing_records[record_key]
                        
                        # Check if any values have changed
                        values_changed = (
                            abs(existing_record.totalEnergyProduced - new_totalEnergyProduced) > 0.001 or
                            abs(existing_record.energyProduced - new_energy_produced) > 0.001 or
                            existing_record.errorCount != new_error_count
                        )
                        
                        if values_changed:
                            # Update existing record with new values
                            existing_record.totalEnergyProduced = new_totalEnergyProduced
                            existing_record.energyProduced = new_energy_produced
                            existing_record.errorCount = new_error_count
                            file_stats['updated'] += 1
                        else:
                            file_stats['unchanged'] += 1
                    else:
                        # Add new record
                        new_record = FactSolarProduction(
                            id_date=date_id,
                            id_time=time_id,
                            id_inverter=inverter_id,
                            id_status=status_id,
                            totalEnergyProduced=new_totalEnergyProduced,
                            energyProduced=new_energy_produced,
                            errorCount=new_error_count
                        )
                        session.add(new_record)
                        file_stats['inserted'] += 1
                        
                        # Add to existing records dictionary
                        existing_records[record_key] = new_record
                    
                except Exception as row_error:
                    error_trace = traceback.format_exc()
                    log_error("Row Processing", f"Error processing row in file {file_name}: {row_error}", error_trace)
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
            
            # Commit any remaining changes
            try:
                session.commit()
                logging.info(f"Committed final batch for file {file_name}")
            except Exception as batch_error:
                error_trace = traceback.format_exc()
                log_error("Final Batch Commit", f"Error during final batch commit: {batch_error}", error_trace)
                session.rollback()
            
            # File complete message (without showing statistics)
            logging.info(f"File complete: {file_name}")
            logging.info(f"Rows processed: {file_stats['processed']}")
            logging.info(f"Rows inserted: {file_stats['inserted']}")
            logging.info(f"Rows updated: {file_stats['updated']}")
            logging.info(f"Rows unchanged: {file_stats['unchanged']}")
            
        except Exception as file_error:
            error_trace = traceback.format_exc()
            log_error("File Processing", f"Error processing {file_type} file {file_name}: {file_error}", error_trace)
            continue
    
    # Update total statistics
    stats['processed'] += file_stats['processed']
    stats['inserted'] += file_stats['inserted']
    stats['updated'] += file_stats['updated']
    stats['unchanged'] += file_stats['unchanged']
    
    # Only show the summary at the end of processing all files
    logging.info(f"=== {file_type} Files Summary ===")
    logging.info(f"Rows Processed: {file_stats['processed']}")
    logging.info(f"Rows Inserted: {file_stats['inserted']}")
    logging.info(f"Rows Updated: {file_stats['updated']}")
    logging.info(f"Rows Unchanged: {file_stats['unchanged']}")

def populate_dim_tables_and_facts():
    """Main ETL process"""
    # Setup logging for the ETL process
    setup_logging("Solarlogs ETL")
    logging.info("Starting Solarlogs ETL process")
    
    # First ensure tables exist
    try:
        init_db()
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Database Init", f"Error initializing database: {str(e)}", error_trace)
        return
        
    session = get_session()
    if not session:
        log_error("Database Session", "Failed to create database session")
        return
    
    try:
        # Get files by category from the get_files_by_category function
        files_by_category = get_files_by_category()
        
        if "Solarlogs" not in files_by_category or not files_by_category["Solarlogs"]:
            log_error("File Search", "No Solarlogs files found in any category")
            return
        
        # Process files for each category (PV and min) directly from files_by_category
        solarlogs_files = files_by_category["Solarlogs"]
        
        # Preload dimension data for efficiency
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
        
        # Fetch all inverters from dimension table
        all_inverters = {}
        for inverter in session.query(DimInverter).all():
            all_inverters[inverter.inverterName] = inverter.id_inverter
        logging.info(f"Preloaded {len(all_inverters)} inverters")
        
        # Fetch all statuses from dimension table
        all_statuses = {}
        for status in session.query(DimStatus).all():
            all_statuses[status.statusName] = status.id_status
        logging.info(f"Preloaded {len(all_statuses)} statuses")
        
        # Create a dictionary for existing records to enable updates
        logging.info("Preloading existing records for updating...")
        existing_records = {}
        for record in session.query(FactSolarProduction).all():
            record_key = (record.id_date, record.id_time, record.id_inverter, record.id_status)
            existing_records[record_key] = record
        
        logging.info(f"Preloaded {len(existing_records)} existing records for checking")
        
        # Process files with stats tracking
        stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'unchanged': 0}
        
        # Process PV files
        if "PV" in solarlogs_files and solarlogs_files["PV"]:
            pv_files = solarlogs_files["PV"]  # These are full paths
            logging.info(f"Processing {len(pv_files)} PV files")
            
            for file_path in pv_files:
                folder = os.path.dirname(file_path)
                filename = os.path.basename(file_path)
                process_files_by_type(session, folder, [filename], "PV", 
                                     all_dates, all_times, all_inverters, 
                                     all_statuses, existing_records, stats)
        else:
            logging.info("No PV files to process")
            
        # Process min files
        if "min" in solarlogs_files and solarlogs_files["min"]:
            min_files = solarlogs_files["min"]  # These are full paths
            logging.info(f"Processing {len(min_files)} min files")
            
            for file_path in min_files:
                folder = os.path.dirname(file_path)
                filename = os.path.basename(file_path)
                process_files_by_type(session, folder, [filename], "min", 
                                     all_dates, all_times, all_inverters, 
                                     all_statuses, existing_records, stats)
        else:
            logging.info("No min files to process")
        
        logging.info("=== Final Summary ===")
        logging.info(f"Total Files Processed: {len(files_by_category['Solarlogs'])}")
        logging.info(f"Total Processed: {stats['processed']}")
        logging.info(f"Total Inserted: {stats['inserted']}")
        logging.info(f"Total Updated: {stats['updated']}")
        logging.info(f"Total Unchanged: {stats['unchanged']}")
        logging.info("===================")
                
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("ETL Process", f"ETL Error: {str(e)}", error_trace)
        session.rollback()
    finally:
        session.close()
        
        # Send error summary if any errors occurred
        send_error_summary("Solarlogs ETL")
        
        logging.info("Solarlogs ETL process completed")

def process_files_from_category(session, folder_path, files, category):
    """Process files from a specific category using the appropriate processor"""
    try:
        logging.info(f"Processing {category} files...")
        
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
        
        # Fetch all inverters from dimension table
        all_inverters = {}
        for inverter in session.query(DimInverter).all():
            all_inverters[inverter.inverterName] = inverter.id_inverter
        logging.info(f"Preloaded {len(all_inverters)} inverters")
        
        # Fetch all statuses from dimension table
        all_statuses = {}
        for status in session.query(DimStatus).all():
            all_statuses[status.statusName] = status.id_status
        logging.info(f"Preloaded {len(all_statuses)} statuses")
        
        # Create a dictionary for existing records to enable updates
        logging.info("Preloading existing records for updating...")
        existing_records = {}
        for record in session.query(FactSolarProduction).all():
            record_key = (record.id_date, record.id_time, record.id_inverter, record.id_status)
            existing_records[record_key] = record
        
        logging.info(f"Preloaded {len(existing_records)} existing records for checking")
        
        # Process files with stats tracking
        stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'unchanged': 0}
        process_files_by_type(session, folder_path, files, category, all_dates, all_times, 
                             all_inverters, all_statuses, existing_records, stats)
        
        return True
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Category Processing", f"Error processing {category} files: {str(e)}", error_trace)
        return False

if __name__ == "__main__":
    populate_dim_tables_and_facts()
