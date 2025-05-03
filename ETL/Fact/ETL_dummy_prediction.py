# filepath: c:\Users\david\OneDrive - HESSO\Git\data_cycle\ETL\Fact\ETL_dummy_prediction.py
import os
import pandas as pd
import logging
import traceback
from datetime import datetime, timedelta
from sqlalchemy.exc import SQLAlchemyError

import sys
# Add parent directory to path to make ETL a proper package import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ETL.utils.logging_utils import log_error, setup_logging, send_error_summary
from ETL.db.base import get_session, init_db
from ETL.db.models import FactPrediction
from ETL.Dim.DimDate import get_or_create_date
from ETL.Dim.DimTime import get_or_create_time
from config import Config

def generate_dummy_prediction_data(start_date=None, days=1):
    """
    Generate dummy prediction data for the specified number of days.
    
    Args:
        start_date: The start date for predictions (default: today)
        days: Number of days to generate data for (default: 1)
    
    Returns:
        tuple: (consumption_df, production_df) with dummy prediction data
    """
    # Use today's date if no start_date provided
    if start_date is None:
        start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
        
    # Set the year to 2023 while keeping current month and days
    start_date = start_date.replace(year=2023)
    
    logging.info(f"Generating dummy prediction data starting from {start_date.strftime('%Y-%m-%d')} for {days} days")
    
    # Generate timestamps for every hour for the specified number of days
    timestamps = []
    current_date = start_date
    
    for day in range(days):
        for hour in range(24):
            timestamps.append(current_date.replace(hour=hour))
        current_date += timedelta(days=1)
    
    # Create consumption DataFrame with dummy value of 10
    cons_df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_consumption': [10.0] * len(timestamps)
    })
    
    # Create production DataFrame with dummy value of 2
    prod_df = pd.DataFrame({
        'timestamp': timestamps,
        'predicted_production': [2.0] * len(timestamps)
    })
    
    # Format datetime for the ETL process
    cons_df['Date'] = cons_df['timestamp'].dt.strftime('%Y-%m-%d')
    cons_df['Time'] = cons_df['timestamp'].dt.strftime('%H:%M:%S')
    cons_df = cons_df[['Date', 'Time', 'predicted_consumption']].rename(columns={'predicted_consumption': 'Value'})
    
    prod_df['Date'] = prod_df['timestamp'].dt.strftime('%Y-%m-%d')
    prod_df['Time'] = prod_df['timestamp'].dt.strftime('%H:%M:%S')
    prod_df = prod_df[['Date', 'Time', 'predicted_production']].rename(columns={'predicted_production': 'Value'})
    
    return cons_df, prod_df

def process_predictions(session, consumption_df, production_df):
    """Processes prediction dataframes and inserts into the database."""
    try:
        stats = {'processed': 0, 'inserted': 0, 'duplicates': 0, 'invalid': 0}

        # --- Preloading Existing Predictions for Duplicate Check ---
        logging.info("Preloading existing prediction records for duplicate checking...")
        existing_records = set((record.id_date, record.id_time) for record in session.query(FactPrediction.id_date, FactPrediction.id_time).all())
        logging.info(f"Preloaded {len(existing_records)} existing prediction records.")

        # --- Merge DataFrames ---
        # Merge consumption and production predictions based on Date and Time
        if consumption_df is not None and production_df is not None:
            merged_df = pd.merge(consumption_df, production_df, on=['Date', 'Time'], how='outer', suffixes=('_cons', '_prod'))
            logging.info(f"Merged consumption and production data. Resulting rows: {len(merged_df)}")
        elif consumption_df is not None:
            merged_df = consumption_df.rename(columns={'Value': 'Value_cons'})
            merged_df['Value_prod'] = None
            logging.info("Processing only consumption predictions.")
        elif production_df is not None:
            merged_df = production_df.rename(columns={'Value': 'Value_prod'})
            merged_df['Value_cons'] = None
            logging.info("Processing only production predictions.")
        else:
            log_error("Processing", "No valid prediction dataframes to process.")
            return stats

        # --- Process Merged Data ---
        batch_size = 1000
        records_to_insert = []

        for _, row in merged_df.iterrows():
            stats['processed'] += 1
            try:
                date_str = row['Date']
                time_str = row['Time']

                # Get or Create Date Dimension ID
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    date_id = get_or_create_date(session, date_obj)
                    if not date_id:
                        log_error("Dimension Creation", f"Failed to get or create Date dimension for {date_str}")
                        stats['invalid'] += 1
                        continue
                except ValueError:
                    log_error("Dimension Creation", f"Invalid date format encountered: {date_str}")
                    stats['invalid'] += 1
                    continue

                # Get or Create Time Dimension ID
                try:
                    time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
                    time_id = get_or_create_time(session, time_obj)
                    if not time_id:
                        log_error("Dimension Creation", f"Failed to get or create Time dimension for {time_str}")
                        stats['invalid'] += 1
                        continue
                except ValueError:
                    log_error("Dimension Creation", f"Invalid time format encountered: {time_str}")
                    stats['invalid'] += 1
                    continue

                # --- Duplicate Check ---
                record_key = (date_id, time_id)
                if record_key in existing_records:
                    stats['duplicates'] += 1
                    continue

                # --- Prepare Data for Insertion ---
                predicted_consumption = row.get('Value_cons')
                predicted_production = row.get('Value_prod')

                # Convert to float, allowing None if NaN
                cons_val = None if pd.isna(predicted_consumption) else float(predicted_consumption)
                prod_val = None if pd.isna(predicted_production) else float(predicted_production)

                records_to_insert.append({
                    'id_date': date_id,
                    'id_time': time_id,
                    'predicted_consumption': cons_val,
                    'predicted_production': prod_val
                })
                stats['inserted'] += 1
                existing_records.add(record_key)

                # --- Batch Commit ---
                if len(records_to_insert) >= batch_size:
                    try:
                        session.bulk_insert_mappings(FactPrediction, records_to_insert)
                        session.commit()
                        logging.info(f"Committed batch of {len(records_to_insert)} prediction records (total processed: {stats['processed']})")
                        records_to_insert = []
                    except Exception as batch_error:
                        error_trace = traceback.format_exc()
                        log_error("Batch Commit", f"Error during prediction batch commit: {batch_error}", error_trace)
                        session.rollback()
                        records_to_insert = []

            except Exception as row_error:
                error_trace = traceback.format_exc()
                log_error("Row Processing", f"Error processing prediction row: {row_error}", error_trace)
                stats['invalid'] += 1
                continue

        # --- Commit Final Batch ---
        if records_to_insert:
            try:
                session.bulk_insert_mappings(FactPrediction, records_to_insert)
                session.commit()
                logging.info(f"Committed final batch of {len(records_to_insert)} prediction records")
            except Exception as batch_error:
                error_trace = traceback.format_exc()
                log_error("Final Batch Commit", f"Error during final prediction batch commit: {batch_error}", error_trace)
                session.rollback()

        # --- Log Summary ---
        logging.info("=== Dummy Prediction ETL Summary ===")
        logging.info(f"Total Processed Rows: {stats['processed']}")
        logging.info(f"Total Inserted Records: {stats['inserted']}")
        logging.info(f"Total Duplicates Skipped: {stats['duplicates']}")
        logging.info(f"Total Invalid Rows Skipped: {stats['invalid']}")
        logging.info("===================================")

        return stats

    except SQLAlchemyError as e:
        error_trace = traceback.format_exc()
        log_error("Database", f"Database Error during prediction processing: {str(e)}", error_trace)
        session.rollback()
        raise
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Processing", f"General Error during prediction processing: {str(e)}", error_trace)
        session.rollback()
        raise


def run_dummy_prediction_etl(start_date=None, days_to_generate=1):
    """
    Main ETL process for generating and loading dummy prediction data.
    
    Args:
        start_date: The date to start generating predictions (default: today)
        days_to_generate: Number of days to generate predictions for (default: 1)
    """
    setup_logging("Dummy Prediction ETL")
    logging.info("Starting Dummy Prediction ETL process")
    
    # --- Initialize Database ---
    try:
        init_db()
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Database Init", f"Error initializing database: {str(e)}", error_trace)
        send_error_summary("Dummy Prediction ETL")
        return

    session = get_session()
    if not session:
        log_error("Database Session", "Failed to create database session")
        send_error_summary("Dummy Prediction ETL")
        return

    try:
        # --- Generate Dummy Data ---
        if start_date is None:
            start_date = datetime.now()
        
        # Set the year to 2023 while preserving current day and month
        start_date = start_date.replace(year=2023)
            
        logging.info(f"Generating dummy data for {days_to_generate} days starting from {start_date.strftime('%Y-%m-%d')}")
        consumption_df, production_df = generate_dummy_prediction_data(start_date, days_to_generate)
        
        # --- Process Generated Data ---
        process_predictions(session, consumption_df, production_df)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("ETL Process", f"Unhandled Dummy Prediction ETL Error: {str(e)}", error_trace)
        if session:
            session.rollback()
    finally:
        if session:
            session.close()
            logging.info("Database session closed.")

        # Send error summary if any errors were logged during the process
        send_error_summary("Dummy Prediction ETL")
        
        logging.info("Dummy Prediction ETL process completed.")


if __name__ == "__main__":
    # Default: Generate predictions for today only (with year 2023)
    run_dummy_prediction_etl()
    
    # Example of generating predictions for multiple days:
    # Custom start date (e.g., tomorrow) and multiple days
    # Note: Year will always be set to 2023 regardless of the input date
    # tomorrow = datetime.now() + timedelta(days=1)
    # run_dummy_prediction_etl(start_date=tomorrow, days_to_generate=7)
