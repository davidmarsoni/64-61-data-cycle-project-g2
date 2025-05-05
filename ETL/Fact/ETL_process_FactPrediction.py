# Required library imports
import os
import pandas as pd
import logging
import traceback
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
from ETL.utils.logging_utils import log_error, setup_logging, send_error_summary
from ETL.db.base import get_session, init_db
from ETL.db.models import FactPrediction
from ETL.Dim.DimDate import get_or_create_date
from ETL.Dim.DimTime import get_or_create_time
from config import Config
from ETL.db.models import DimDate, DimTime

def get_prediction_files():
    """Get prediction files from the api_results directory"""
    prediction_files = {
        'consumption': [],
        'production': []
    }
    
    # Calculate absolute path to api_results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels from ETL/Fact
    api_results_dir = os.path.join(project_root, 'machineLearning', 'api_results')
    
    print(f"Looking for prediction files in: {api_results_dir}")
    
    if not os.path.exists(api_results_dir):
        os.makedirs(api_results_dir, exist_ok=True)
        log_error("File Search", f"API Results directory created: {api_results_dir}")
        return prediction_files
    
    # Find consumption and production files
    for filename in os.listdir(api_results_dir):
        if filename.endswith('.csv'):
            file_path = os.path.join(api_results_dir, filename)
            
            if filename.startswith('consumption_'):
                prediction_files['consumption'].append(file_path)
            elif filename.startswith('production_'):
                prediction_files['production'].append(file_path)
    
    # Sort by modification time to get the latest files
    prediction_files['consumption'].sort(key=lambda x: os.path.getmtime(x), reverse=True)
    prediction_files['production'].sort(key=lambda x: os.path.getmtime(x), reverse=True)

    if prediction_files['consumption']:
        prediction_files['consumption'] = [prediction_files['consumption'][0]]
    if prediction_files['production']:
        prediction_files['production'] = [prediction_files['production'][0]]
        
    return prediction_files

def validate_prediction_row(row):
    """Validate prediction data row"""
    try:
        # Check if required columns exist
        if 'date' not in row or pd.isna(row['date']):
            return False, "Missing or invalid date"
        
        # Check if at least one of the prediction values exists
        if ('consumptionPrediction' not in row or pd.isna(row['consumptionPrediction'])) and \
           ('productionPrediction' not in row or pd.isna(row['productionPrediction'])):
            return False, "Missing both consumption and production predictions"
        
        return True, ""
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def merge_prediction_dataframes(consumption_df, production_df):
    """Merge consumption and production dataframes on date"""
    if consumption_df is None and production_df is None:
        return None
    
    # Standardize column names and date format
    if consumption_df is not None:
        # Ensure date column is in proper format
        consumption_df['date'] = pd.to_datetime(consumption_df['date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
        # Add hour column (daily consumption is assigned to 00:00)
        consumption_df['hour'] = '00:00'
    
    if production_df is not None:
        # Standardize date format
        production_df['date'] = pd.to_datetime(production_df['date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
    
    # If both dataframes exist, create a combined result
    combined_records = []
    
    # Process consumption data (daily records)
    if consumption_df is not None:
        for _, row in consumption_df.iterrows():
            combined_records.append({
                'date': row['date'],
                'hour': row['hour'],
                'consumptionPrediction': row['consumptionPrediction'],
                'productionPrediction': None
            })
    
    # Process production data (hourly records)
    if production_df is not None:
        for _, row in production_df.iterrows():
            # For hourly production data
            combined_records.append({
                'date': row['date'],
                'hour': row['hour'],
                'consumptionPrediction': None,
                'productionPrediction': row['productionPrediction']
            })
    
    # Convert the list of records to a DataFrame
    if combined_records:
        return pd.DataFrame(combined_records)
    else:
        return None

def process_prediction_files(session):
    """Process prediction files and insert into database"""
    try:
        prediction_files = get_prediction_files()
        stats = {'processed': 0, 'inserted': 0, 'updated': 0, 'skipped': 0}
        
        logging.info(f"Found {len(prediction_files['consumption'])} consumption prediction files and "
                   f"{len(prediction_files['production'])} production prediction files")
        
        # Preload dimension data to reduce database queries
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
        
        # Get existing records to enable updates
        logging.info("Preloading existing records for comparison...")
        existing_records = {}
        for record in session.query(FactPrediction).all():
            record_key = (record.id_date, record.id_time)
            existing_records[record_key] = record
        
        logging.info(f"Preloaded {len(existing_records)} existing prediction records")
        
        # Load latest consumption prediction file if available
        consumption_df = None
        if prediction_files['consumption']:
            latest_consumption_file = prediction_files['consumption'][0]
            logging.info(f"Processing consumption prediction file: {latest_consumption_file}")
            try:
                consumption_df = pd.read_csv(latest_consumption_file)
                logging.info(f"Loaded {len(consumption_df)} consumption predictions")
            except Exception as e:
                error_trace = traceback.format_exc()
                log_error("File Loading", f"Error loading consumption file {latest_consumption_file}: {e}", error_trace)
        
        # Load latest production prediction file if available
        production_df = None
        if prediction_files['production']:
            latest_production_file = prediction_files['production'][0]
            logging.info(f"Processing production prediction file: {latest_production_file}")
            try:
                production_df = pd.read_csv(latest_production_file)
                logging.info(f"Loaded {len(production_df)} production predictions")
            except Exception as e:
                error_trace = traceback.format_exc()
                log_error("File Loading", f"Error loading production file {latest_production_file}: {e}", error_trace)
        
        # Merge the prediction dataframes
        merged_df = merge_prediction_dataframes(consumption_df, production_df)
        
        if merged_df is None or merged_df.empty:
            log_error("Data Merging", "No valid prediction data found after merging")
            return stats
        
        logging.info(f"Processing {len(merged_df)} merged prediction records")
        
        # NEW APPROACH: Only delete predictions for dates that will be replaced
        # Get the dates covered in your new prediction files
        new_prediction_dates = set(merged_df['date'].unique())
        
        # Only delete predictions for dates that will be replaced
        if new_prediction_dates:
            # Convert to list of date IDs
            new_date_ids = []
            for date_str in new_prediction_dates:
                if date_str in all_dates:
                    new_date_ids.append(all_dates[date_str])
                else:
                    # If date not in dimension table, log it
                    logging.info(f"Date {date_str} not found in dimension table, will be created during processing")
            
            # Delete only the predictions for these specific dates
            if new_date_ids:
                deleted_count = session.query(FactPrediction).filter(
                    FactPrediction.id_date.in_(new_date_ids)
                ).delete(synchronize_session='fetch')
                
                session.commit()
                logging.info(f"Deleted {deleted_count} predictions for dates that will be replaced with new predictions")
                
                # Remove deleted records from existing_records dictionary
                for key in list(existing_records.keys()):
                    if key[0] in new_date_ids:
                        del existing_records[key]
        
        # Create dimension mappings
        mappings = {
            'date': {},
            'time': {}
        }
        
        # Batch processing
        batch_size = 1000
        batch_counter = 0
        
        # Process each row in the merged dataframe
        for _, row in merged_df.iterrows():
            stats['processed'] += 1
            
            try:
                # Validate the row
                is_valid, validation_message = validate_prediction_row(row)
                if not is_valid:
                    logging.warning(f"Skipping invalid row: {validation_message}")
                    stats['skipped'] += 1
                    continue
                
                # Extract date and convert to proper format
                date_str = row['date']  # Should already be YYYY-MM-DD format
                if '-' not in date_str:
                    # Convert from DD.MM.YYYY to YYYY-MM-DD if necessary
                    date_parts = date_str.split('.')
                    if len(date_parts) == 3:
                        date_str = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                
                # Get or create date_id
                if date_str not in mappings['date']:
                    if date_str in all_dates:
                        mappings['date'][date_str] = all_dates[date_str]
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        date_id = get_or_create_date(session, date_obj)
                        mappings['date'][date_str] = date_id
                        all_dates[date_str] = date_id
                date_id = mappings['date'][date_str]
                
                # Extract time and convert to proper format
                time_str = row['hour'] if 'hour' in row else '00:00'
                # Ensure it's in HH:MM:SS format
                if len(time_str.split(':')) == 2:
                    time_str = f"{time_str}:00"
                
                # Get or create time_id
                if time_str not in mappings['time']:
                    if time_str in all_times:
                        mappings['time'][time_str] = all_times[time_str]
                    else:
                        time_obj = datetime.strptime(time_str, '%H:%M:%S')
                        time_id = get_or_create_time(session, time_obj)
                        mappings['time'][time_str] = time_id
                        all_times[time_str] = time_id
                time_id = mappings['time'][time_str]
                
                # Get prediction values (might be missing for some records)
                consumption_value = float(row['consumptionPrediction']) if 'consumptionPrediction' in row and pd.notna(row['consumptionPrediction']) else None
                production_value = float(row['productionPrediction']) if 'productionPrediction' in row and pd.notna(row['productionPrediction']) else None
                
                # Check if record exists and needs updating
                record_key = (date_id, time_id)
                
                if record_key in existing_records:
                    # Get existing record
                    existing_record = existing_records[record_key]
                    
                    # Determine what needs updating
                    changes_needed = []
                    
                    if consumption_value is not None and (
                        existing_record.predicted_consumption is None or 
                        abs(existing_record.predicted_consumption - consumption_value) > 0.001
                    ):
                        existing_record.predicted_consumption = consumption_value
                        changes_needed.append('consumption')
                    
                    if production_value is not None and (
                        existing_record.predicted_production is None or 
                        abs(existing_record.predicted_production - production_value) > 0.001
                    ):
                        existing_record.predicted_production = production_value
                        changes_needed.append('production')
                    
                    if changes_needed:
                        stats['updated'] += 1
                        logging.debug(f"Updated {', '.join(changes_needed)} for {date_str} {time_str}")
                    else:
                        logging.debug(f"No updates needed for {date_str} {time_str}")
                else:
                    # Create a new record
                    new_record = FactPrediction(
                        id_date=date_id,
                        id_time=time_id,
                        predicted_consumption=consumption_value,
                        predicted_production=production_value
                    )
                    session.add(new_record)
                    existing_records[record_key] = new_record
                    stats['inserted'] += 1
                    logging.debug(f"Inserted new record for {date_str} {time_str}")
                
                # Commit every batch_size records
                batch_counter += 1
                if batch_counter >= batch_size:
                    session.commit()
                    logging.info(f"Committed batch of {batch_counter} records")
                    batch_counter = 0
            
            except Exception as row_error:
                error_trace = traceback.format_exc()
                log_error("Row Processing", f"Error processing prediction row: {row_error}", error_trace)
                stats['skipped'] += 1
                continue
        
        # Commit any remaining changes
        if batch_counter > 0:
            try:
                session.commit()
                logging.info(f"Committed final batch of {batch_counter} records")
            except Exception as batch_error:
                error_trace = traceback.format_exc()
                log_error("Batch Commit", f"Error during final batch commit: {batch_error}", error_trace)
                session.rollback()
        
        logging.info("=== Prediction ETL Summary ===")
        logging.info(f"Records Processed: {stats['processed']}")
        logging.info(f"Records Inserted: {stats['inserted']}")
        logging.info(f"Records Updated: {stats['updated']}")
        logging.info(f"Records Skipped: {stats['skipped']}")
        logging.info("=============================")
        
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
    """Main ETL process for populating FactPrediction table"""
    # Setup logging
    setup_logging("Prediction ETL")
    logging.info("Starting Prediction ETL process")
    
    # Initialize database
    try:
        init_db()
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Database Init", f"Error initializing database: {str(e)}", error_trace)
        send_error_summary("Prediction ETL")
        return
    
    # Create session
    session = get_session()
    if not session:
        log_error("Database Session", "Failed to create database session")
        send_error_summary("Prediction ETL")
        return
    
    try:
        # Process the prediction files
        process_prediction_files(session)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("ETL Process", f"ETL Error: {str(e)}", error_trace)
        session.rollback()
    finally:
        session.close()
        
        # Send error summary if there were any errors
        send_error_summary("Prediction ETL")
        
        logging.info("Prediction ETL process completed")

if __name__ == "__main__":
    populate_dim_tables_and_facts()