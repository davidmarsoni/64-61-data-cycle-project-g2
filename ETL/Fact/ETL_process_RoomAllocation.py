# Required library imports
import os
import sys
import pandas as pd
import logging
import traceback
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

from ETL.utils.logging_utils import log_error, setup_logging, send_error_summary
from ETL.db.base import get_session, init_db
from ETL.db.models import FactBookings
from ETL.Dim.DimDate import get_or_create_date
from ETL.Dim.DimTime import get_or_create_time
from ETL.Dim.DimRoom import get_or_create_room
from ETL.Dim.DimUser import get_or_create_user
from ETL.Dim.DimActivity import get_or_create_activity
from ETL.Dim.DimBookingType import get_or_create_booking_type
from ETL.Dim.DimDivision import get_or_create_division
from ETL.Dim.DimClassroom import get_or_create_classroom
from config import Config
from ETL.db.models import (
    DimDate, DimTime, DimRoom, DimUser, 
    DimActivity, DimBookingType, DimDivision, DimClassroom
)

SUBFOLDERS = {
    "BellevueBooking": "RoomAllocations",
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
    
    # Process BellevueBooking folder
    main_folder_path = os.path.join(actual_data_dir, "BellevueBooking")
    if not os.path.exists(main_folder_path):
        log_error("File Search", f"Main folder path does not exist: {main_folder_path}")
        return files_by_category
        
    files_by_category["BellevueBooking"] = {}
    
    # Get all CSV files for room allocations
    for f in os.listdir(main_folder_path):
        if f.endswith('.csv'):
            category = SUBFOLDERS["BellevueBooking"]
            if category not in files_by_category["BellevueBooking"]:
                files_by_category["BellevueBooking"][category] = []
            files_by_category["BellevueBooking"][category].append(os.path.join(main_folder_path, f))
            logging.info(f"Found {category} file: {f}")
    
    return files_by_category

def validate_room_allocation_row(room_name, date_val, start_time, end_time, user_name):
    """Validate a single row of room allocation data"""
    try:
        # Check if all required values exist
        required_fields = [room_name, date_val, start_time, end_time, user_name]
        if any(val is None or pd.isna(val) for val in required_fields):
            return False, "Missing required values"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_room_allocation_file(session, allocation_df):
    """Process room allocation file and insert into database using ORM"""
    try:
        # Add tracking for deactivated reservations
        stats = {'processed': 0, 'inserted': 0, 'duplicates': 0, 'invalid': 0, 'deactivated': 0}
        
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
        
        # Fetch all rooms from dimension table
        all_rooms = {}
        for room in session.query(DimRoom).all():
            all_rooms[room.room_name] = room.id_room
        logging.info(f"Preloaded {len(all_rooms)} rooms")
        
        # Fetch all users from dimension table
        all_users = {}
        for user in session.query(DimUser).all():
            all_users[user.username.lower()] = user.id_user
        logging.info(f"Preloaded {len(all_users)} users")
        
        # Fetch all activities from dimension table
        all_activities = {}
        for activity in session.query(DimActivity).all():
            all_activities[activity.activity_name.lower()] = activity.id_activity
        logging.info(f"Preloaded {len(all_activities)} activities")
        
        # Fetch all booking types from dimension table
        all_booking_types = {}
        for booking_type in session.query(DimBookingType).all():
            key = (booking_type.code, booking_type.booking_type)
            all_booking_types[key] = booking_type.id_booking_type 
        logging.info(f"Preloaded {len(all_booking_types)} booking types")
        
        # Fetch all classrooms from dimension table
        all_classrooms = {}
        for classroom in session.query(DimClassroom).all():
            all_classrooms[classroom.classroom_name.lower()] = classroom.id_classroom
        logging.info(f"Preloaded {len(all_classrooms)} classrooms")
        
        # Create or ensure a default booking type exists
        default_booking_type_code = "DEF"
        default_booking_type_name = "Default"
        default_key = (default_booking_type_code, default_booking_type_name)
        
        if default_key not in all_booking_types:
            default_booking_type_id = get_or_create_booking_type(session, default_booking_type_code, default_booking_type_name)
            all_booking_types[default_key] = default_booking_type_id
            logging.info(f"Created default booking type with ID: {default_booking_type_id}")
        else:
            logging.info(f"Using existing default booking type with ID: {all_booking_types[default_key]}")
        
        # Fetch all divisions from dimension table
        all_divisions = {}
        for division in session.query(DimDivision).all():
            all_divisions[division.division_name.lower()] = division.id_division
        logging.info(f"Preloaded {len(all_divisions)} divisions")
        
        # =======================================
        # STEP 1: Create identifiers for all current file reservations
        # =======================================
        current_file_reservations = set()
        
        logging.info("Creating identifiers for current file reservations...")
        for _, row in allocation_df.iterrows():
            try:
                # Extract required fields
                date_str = row['Date']
                time_start = row['start_time'] 
                time_end = row['end_time']
                room_name = row['room_name']
                user_name = row['user_name']
                
                # Fix missing user_name
                if pd.isna(user_name):
                    user_name = "Unknown User"
                    
                # Create a unique identifier for this reservation
                reservation_id = f"{date_str}_{time_start}_{time_end}_{room_name}_{user_name}"
                current_file_reservations.add(reservation_id)
                
            except Exception as e:
                continue
                
        logging.info(f"Found {len(current_file_reservations)} reservations in current file")
                
        # =======================================
        # STEP 2: Get all active reservations from the database for relevant dates
        # =======================================
        date_set = set()
        for date_str in allocation_df['Date'].unique():
            if not pd.isna(date_str) and date_str in all_dates:
                date_set.add(all_dates[date_str])
                
        logging.info(f"Fetching active reservations for {len(date_set)} dates...")
        
        # Query active reservations for these dates
        active_reservations = {}
        if date_set:
            booking_query = session.query(
                FactBookings.id_booking,
                FactBookings.id_date,
                FactBookings.id_time_start,
                FactBookings.id_time_end,
                FactBookings.id_room,
                FactBookings.id_user,
                FactBookings.external_id
            ).filter(
                FactBookings.id_date.in_(list(date_set)),
                FactBookings.is_active == True
            ).all()
            
            for booking in booking_query:
                # Use external_id for the key if it exists, otherwise skip
                if booking.external_id:
                    active_reservations[booking.external_id] = booking.id_booking
                    
            logging.info(f"Found {len(active_reservations)} active reservations in database")
        
        # Create a set for duplicate checking
        existing_records = set()
        for record in session.query(
            FactBookings.id_date,
            FactBookings.id_time_start,
            FactBookings.id_time_end,
            FactBookings.id_room,
            FactBookings.id_user
        ).filter(FactBookings.is_active == True).all():
            existing_records.add((
                record.id_date, 
                record.id_time_start,
                record.id_time_end,
                record.id_room,
                record.id_user
            ))
        
        # =======================================
        # STEP 3: Process current reservations and add new ones
        # =======================================
        batch_size = 1000
        records_to_insert = []
        
        # Process each row in allocation dataframe
        for _, row in allocation_df.iterrows():
            stats['processed'] += 1
            
            try:
                # Extract and validate required fields
                date_str = row['Date']
                time_start = row['start_time']
                time_end = row['end_time']
                room_name = row['room_name']
                user_name = row['user_name']
                
                # Fix missing user_name
                if pd.isna(user_name):
                    user_name = "Unknown User"
                    
                # Create unique identifier for tracking
                reservation_externalId = f"{date_str}_{time_start}_{time_end}_{room_name}_{user_name}"
                
                # Skip if already in active reservations
                if reservation_externalId in active_reservations:
                    stats['duplicates'] += 1
                    continue
                    
                # Extract optional fields
                professor = row.get('professor')
                activity = row.get('activity')
                booking_type_code = row.get('codes')
                booking_type_name = row.get('reservation_type')
                division = row.get('division')
                classroom = row.get('class') 
                
                # Validate required data
                is_valid, message = validate_room_allocation_row(
                    room_name, date_str, time_start, time_end, user_name
                )
                if not is_valid:
                    stats['invalid'] += 1
                    log_error("Data Validation", 
                            f"Invalid row: {message} - Date: {date_str}, "
                            f"Time: {time_start}-{time_end}, Room: {room_name}, "
                            f"User: {user_name}")
                    continue
                
                # Get dimension IDs for required fields
                if date_str not in all_dates:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    date_id = get_or_create_date(session, date_obj)
                    all_dates[date_str] = date_id
                date_id = all_dates[date_str]
                
                # Handle time format and add seconds if missing
                if time_start and ':' in time_start and time_start.count(':') == 1:
                    time_start = f"{time_start}:00"
                if time_end and ':' in time_end and time_end.count(':') == 1:
                    time_end = f"{time_end}:00"
                
                if time_start not in all_times:
                    try:
                        time_obj = datetime.strptime(time_start, '%H:%M:%S')
                        time_id = get_or_create_time(session, time_obj)
                        all_times[time_start] = time_id
                    except ValueError:
                        time_obj = datetime.strptime(time_start, '%H:%M')
                        time_id = get_or_create_time(session, time_obj)
                        all_times[time_start] = time_id
                time_start_id = all_times[time_start]
                
                if time_end not in all_times:
                    try:
                        time_obj = datetime.strptime(time_end, '%H:%M:%S')
                        time_id = get_or_create_time(session, time_obj)
                        all_times[time_end] = time_id
                    except ValueError:
                        time_obj = datetime.strptime(time_end, '%H:%M')
                        time_id = get_or_create_time(session, time_obj)
                        all_times[time_end] = time_id
                time_end_id = all_times[time_end]
                
                if room_name not in all_rooms:
                    room_id = get_or_create_room(session, room_name)
                    all_rooms[room_name] = room_id
                room_id = all_rooms[room_name]
                
                if user_name.lower() not in all_users:
                    user_id = get_or_create_user(session, user_name)
                    all_users[user_name.lower()] = user_id
                user_id = all_users[user_name.lower()]
                
                # Get dimension IDs for optional fields
                professor_id = None
                if professor and not pd.isna(professor):
                    if professor.lower() not in all_users:
                        professor_id = get_or_create_user(session, professor)
                        all_users[professor.lower()] = professor_id
                    professor_id = all_users[professor.lower()]
                
                activity_id = None
                if activity and not pd.isna(activity):
                    if activity.lower() not in all_activities:
                        activity_id = get_or_create_activity(session, activity)
                        all_activities[activity.lower()] = activity_id
                    activity_id = all_activities[activity.lower()]
                
                booking_type_id = None
                if booking_type_code and booking_type_name and not pd.isna(booking_type_code) and not pd.isna(booking_type_name): # Used renamed variable
                    key = (booking_type_code, booking_type_name)
                    if key not in all_booking_types:
                        booking_type_id = get_or_create_booking_type(session, booking_type_code, booking_type_name) # Used renamed variable
                        all_booking_types[key] = booking_type_id
                    else:
                        booking_type_id = all_booking_types[key]
                else:
                    # Use default booking type if none provided
                    booking_type_id = all_booking_types[default_key]
                
                division_id = None
                if division and not pd.isna(division):
                    if division.lower() not in all_divisions:
                        division_id = get_or_create_division(session, division)
                        all_divisions[division.lower()] = division_id
                    division_id = all_divisions[division.lower()]
                
                classroom_id = None
                if classroom and not pd.isna(classroom):
                    if classroom.lower() not in all_classrooms:
                        classroom_id = get_or_create_classroom(session, classroom)
                        all_classrooms[classroom.lower()] = classroom_id
                    classroom_id = all_classrooms[classroom.lower()]
                
                # Check for duplicates in memory
                record_key = (date_id, time_start_id, time_end_id, room_id, user_id)
                if record_key in existing_records:
                    stats['duplicates'] += 1
                    continue
                
                # Add to batch for insertion
                # Convert the FactBookings object attributes to a dictionary for bulk_insert_mappings
                booking_data = {
                    'id_date': date_id,
                    'id_time_start': time_start_id,
                    'id_time_end': time_end_id,
                    'id_room': room_id,
                    'id_user': user_id,
                    'id_professor': professor_id,
                    'id_classroom': classroom_id,
                    'id_booking_type': booking_type_id,
                    'id_division': division_id,
                    'id_activity': activity_id,
                    'is_active': True,
                    'external_id': reservation_externalId
                }
                records_to_insert.append(booking_data) # Append the dictionary, not the object
                stats['inserted'] += 1
                
                # Add to existing records to prevent future duplicates
                existing_records.add(record_key)
                
                # Batch commit
                if len(records_to_insert) >= batch_size:
                    try:
                        session.bulk_insert_mappings(FactBookings, records_to_insert) # Pass the list of dictionaries
                        session.commit()
                        logging.info(f"Committed batch of {len(records_to_insert)} records (total processed: {stats['processed']})")
                        records_to_insert = []
                    except Exception as batch_error:
                        error_trace = traceback.format_exc()
                        log_error("Batch Commit", f"Error during batch commit: {batch_error}", error_trace)
                        session.rollback()
                        records_to_insert = [] # Clear batch on error
                
            except Exception as row_error:
                error_trace = traceback.format_exc()
                log_error("Row Processing", f"Error processing row: {row_error}", error_trace)
                continue
        
        # Commit any remaining rows in the final batch
        if records_to_insert:
            try:
                session.bulk_insert_mappings(FactBookings, records_to_insert) # Pass the list of dictionaries
                session.commit()
                logging.info(f"Committed final batch of {len(records_to_insert)} records")
            except Exception as batch_error:
                error_trace = traceback.format_exc()
                log_error("Final Batch Commit", f"Error during final batch commit: {batch_error}", error_trace)
                session.rollback()
                
        # =======================================
        # STEP 4: Identify and deactivate removed reservations
        # =======================================
        reservations_to_deactivate = []
        for external_id, booking_id in active_reservations.items():
            if external_id not in current_file_reservations:
                reservations_to_deactivate.append(booking_id)
                
        if reservations_to_deactivate:
            logging.info(f"Found {len(reservations_to_deactivate)} reservations to deactivate")
            
            # Deactivate reservations in batches
            deactivate_batch_size = 500
            for i in range(0, len(reservations_to_deactivate), deactivate_batch_size):
                batch_ids = reservations_to_deactivate[i:i + deactivate_batch_size]
                try:
                    session.query(FactBookings).filter(
                        FactBookings.id_booking.in_(batch_ids)
                    ).update({FactBookings.is_active: False}, synchronize_session=False)
                    session.commit()
                    stats['deactivated'] += len(batch_ids)
                    logging.info(f"Deactivated batch of {len(batch_ids)} reservations")
                except Exception as deactivate_error:
                    error_trace = traceback.format_exc()
                    log_error("Deactivation Batch Commit", f"Error during deactivation batch commit: {deactivate_error}", error_trace)
                    session.rollback()
        
        # Log statistics for this file
        logging.info("=== Summary ===")
        logging.info(f"Total Processed: {stats['processed']}")
        logging.info(f"Total Inserted: {stats['inserted']}")
        logging.info(f"Total Duplicates: {stats['duplicates']}")
        logging.info(f"Total Invalid: {stats['invalid']}")
        logging.info(f"Total Deactivated: {stats['deactivated']}")
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
    setup_logging("Room Allocation ETL")
    logging.info("Starting Room Allocation ETL process")
    
    # Check for files first before attempting any database operations
    organized_files = get_files_by_category()
    
    # If the directory is empty, log an INFO message and exit 
    if not organized_files or not any(organized_files.values()):
        logging.info("No files found in the directory. This is normal if data files don't arrive daily.")
        logging.info("Room Allocation ETL process completed without processing any files")
        return
    
    # Now that we know files exist, proceed with database initialization
    try:
        init_db()
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error("Database Init", f"Error initializing database: {str(e)}", error_trace)
        send_error_summary("Room Allocation ETL")
        return
    
    # Create session after successful db initialization
    session = get_session()
    if not session:
        log_error("Database Session", "Failed to create database session")
        send_error_summary("Room Allocation ETL")
        return
    
    try:
        total_stats = {'processed': 0, 'inserted': 0, 'duplicates': 0, 'invalid': 0}
        
        for main_folder, categories in organized_files.items():
            room_allocation_files = categories.get('RoomAllocations', [])
            
            for allocation_file in room_allocation_files:
                try:
                    logging.info(f"Processing room allocation file: {os.path.basename(allocation_file)}")
                    
                    # Read and process the allocation file
                    allocation_df = pd.read_csv(allocation_file)
                    
                    # Process the files
                    file_stats = process_room_allocation_file(session, allocation_df)
                    
                    # Update total stats
                    for key in total_stats:
                        if key in file_stats:
                            total_stats[key] += file_stats[key]
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    log_error("File Processing", f"Error processing file {allocation_file}: {str(e)}", error_trace)
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
        
        send_error_summary("Room Allocation ETL")
        
        # Log completion message
        logging.info("Room Allocation ETL process completed")

if __name__ == "__main__":
    populate_dim_tables_and_facts()

