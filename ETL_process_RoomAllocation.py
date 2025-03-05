"""
ETL script to populate the FactBookings table and related dimensions
using SQLAlchemy ORM.
"""

import os
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
import logging
import glob
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("room_allocation_etl.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global configuration
BASE_DIR = os.getenv('BASE_DIR', 'C:\\DataCollection')
current_date = datetime.now().strftime('%Y-%m-%d')
CLEAN_DATA_DIR = os.path.join(BASE_DIR, f"cleaned_data_{current_date}")
BOOKING_DATA_DIR = os.path.join(CLEAN_DATA_DIR, "BellevueBooking")

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

class DimRoom(Base):
    __tablename__ = 'DimRoom'
    
    id_room = Column(Integer, primary_key=True, autoincrement=True)
    roomName = Column(String(255), nullable=False)
    roomFullName = Column(String(255), nullable=False)

class DimUser(Base):
    __tablename__ = 'DimUser'
    
    id_user = Column(Integer, primary_key=True, autoincrement=True)
    userName = Column(String(255), nullable=False)

class DimActivity(Base):
    __tablename__ = 'DimActivity'
    
    id_activity = Column(Integer, primary_key=True, autoincrement=True)
    activityName = Column(String(255), nullable=False)

class DimBookingType(Base):
    __tablename__ = 'DimBookingType'
    
    id_bookingType = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(50), nullable=False)
    bookingType = Column(String(100), nullable=False)

class DimDivision(Base):
    __tablename__ = 'DimDivision'
    
    id_division = Column(Integer, primary_key=True, autoincrement=True)
    divisionName = Column(String(255), nullable=False)

class FactBookings(Base):
    __tablename__ = 'FactBookings'
    
    id_booking = Column(Integer, primary_key=True, autoincrement=True)
    id_date = Column(Integer, ForeignKey('DimDate.id_date'), nullable=False)
    id_time_start = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_time_end = Column(Integer, ForeignKey('DimTime.id_time'), nullable=False)
    id_room = Column(Integer, ForeignKey('DimRoom.id_room'), nullable=False)
    id_user = Column(Integer, ForeignKey('DimUser.id_user'), nullable=False)
    id_professor = Column(Integer, ForeignKey('DimUser.id_user'), nullable=True)
    id_bookingType = Column(Integer, ForeignKey('DimBookingType.id_bookingType'), nullable=False)
    id_division = Column(Integer, ForeignKey('DimDivision.id_division'), nullable=True)
    id_activity = Column(Integer, ForeignKey('DimActivity.id_activity'), nullable=True)

class RoomAllocationETL:
    """ETL process for room allocation data"""
    
    def __init__(self):
        """Initialize the ETL process"""
        self.engine = create_engine(CONNECTION_STRING)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Caches for dimension lookups
        self.date_cache = {}
        self.time_cache = {}
        self.room_cache = {}
        self.user_cache = {}
        self.activity_cache = {}
        self.booking_type_cache = {}
        self.division_cache = {}
        
        # Pre-load dimension caches
        self._preload_dimensions()

    def _preload_dimensions(self):
        """Pre-load dimension tables to minimize database queries"""
        logger.info("Pre-loading dimension data...")
        
        # Load dates
        for date_row in self.session.query(DimDate).all():
            date_key = f"{date_row.year}-{date_row.month:02d}-{date_row.day:02d}"
            self.date_cache[date_key] = date_row.id_date
            
        # Load times
        for time_row in self.session.query(DimTime).all():
            time_key = f"{time_row.hour:02d}:{time_row.minute:02d}"
            self.time_cache[time_key] = time_row.id_time
            
        # Load rooms
        for room_row in self.session.query(DimRoom).all():
            self.room_cache[room_row.roomName] = room_row.id_room
            
        # Load users
        for user_row in self.session.query(DimUser).all():
            self.user_cache[user_row.userName.lower()] = user_row.id_user
            
        # Load activities
        for activity_row in self.session.query(DimActivity).all():
            self.activity_cache[activity_row.activityName.lower()] = activity_row.id_activity
            
        # Load booking types
        for booking_type_row in self.session.query(DimBookingType).all():
            self.booking_type_cache[(booking_type_row.code, booking_type_row.bookingType)] = booking_type_row.id_bookingType
            
        # Load divisions
        for division_row in self.session.query(DimDivision).all():
            self.division_cache[division_row.divisionName.lower()] = division_row.id_division
        
        logger.info(f"Loaded {len(self.date_cache)} dates, {len(self.time_cache)} times, "
                   f"{len(self.room_cache)} rooms, {len(self.user_cache)} users, "
                   f"{len(self.activity_cache)} activities, {len(self.booking_type_cache)} booking types, "
                   f"{len(self.division_cache)} divisions")

    def get_or_create_date_id(self, date_obj):
        """Get or create date dimension entry"""
        date_str = date_obj.strftime('%Y-%m-%d')
        
        if date_str in self.date_cache:
            return self.date_cache[date_str]
            
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
        # Convert to HH:MM format (drop seconds if present)
        if ':' in time_str:
            if len(time_str.split(':')) > 1:
                hour, minute = time_str.split(':')[:2]
                time_key = f"{int(hour):02d}:{int(minute):02d}"
            else:
                hour = time_str.split(':')[0]
                time_key = f"{int(hour):02d}:00"
        else:
            # Handle cases where time is given as a number (e.g. "6" for 6:00)
            time_key = f"{int(time_str):02d}:00"
            
        if time_key in self.time_cache:
            return self.time_cache[time_key]
            
        hour, minute = time_key.split(':')
        
        # Check if time exists
        time_entry = self.session.query(DimTime).filter_by(
            hour=int(hour),
            minute=int(minute)
        ).first()
        
        if time_entry:
            self.time_cache[time_key] = time_entry.id_time
            return time_entry.id_time
            
        # Create new time
        new_time = DimTime(
            hour=int(hour),
            minute=int(minute)
        )
        self.session.add(new_time)
        self.session.flush()
        
        self.time_cache[time_key] = new_time.id_time
        return new_time.id_time
        
    def get_or_create_room_id(self, room_name, room_full_name):
        """Get or create room dimension entry"""
        if room_name in self.room_cache:
            return self.room_cache[room_name]
            
        # Check if room exists
        room_entry = self.session.query(DimRoom).filter_by(
            roomName=room_name
        ).first()
        
        if room_entry:
            self.room_cache[room_name] = room_entry.id_room
            return room_entry.id_room
            
        # Create new room
        new_room = DimRoom(
            roomName=room_name,
            roomFullName=room_full_name
        )
        self.session.add(new_room)
        self.session.flush()
        
        self.room_cache[room_name] = new_room.id_room
        return new_room.id_room
        
    def get_or_create_user_id(self, username):
        """Get or create user dimension entry"""
        # Handle None/NaN
        if pd.isna(username) or username is None or username == '':
            return None
            
        username_lower = username.lower()
        
        if username_lower in self.user_cache:
            return self.user_cache[username_lower]
            
        # Check if user exists
        user_entry = self.session.query(DimUser).filter_by(
            userName=username
        ).first()
        
        if user_entry:
            self.user_cache[username_lower] = user_entry.id_user
            return user_entry.id_user
            
        # Create new user
        new_user = DimUser(
            userName=username
        )
        self.session.add(new_user)
        self.session.flush()
        
        self.user_cache[username_lower] = new_user.id_user
        return new_user.id_user
        
    def get_or_create_activity_id(self, activity):
        """Get or create activity dimension entry"""
        # Handle None/NaN
        if pd.isna(activity) or activity is None or activity == '':
            return None
            
        activity_lower = activity.lower()
        
        if activity_lower in self.activity_cache:
            return self.activity_cache[activity_lower]
            
        # Check if activity exists
        activity_entry = self.session.query(DimActivity).filter_by(
            activityName=activity
        ).first()
        
        if activity_entry:
            self.activity_cache[activity_lower] = activity_entry.id_activity
            return activity_entry.id_activity
            
        # Create new activity
        new_activity = DimActivity(
            activityName=activity
        )
        self.session.add(new_activity)
        self.session.flush()
        
        self.activity_cache[activity_lower] = new_activity.id_activity
        return new_activity.id_activity
        
    def get_or_create_booking_type_id(self, code, booking_type):
        """Get or create booking type dimension entry"""
        # Handle None/NaN
        if pd.isna(code) or code is None:
            code = "N/A"
        if pd.isna(booking_type) or booking_type is None:
            booking_type = "N/A"
            
        key = (code, booking_type)
        
        if key in self.booking_type_cache:
            return self.booking_type_cache[key]
            
        # Check if booking type exists
        booking_type_entry = self.session.query(DimBookingType).filter_by(
            code=code,
            bookingType=booking_type
        ).first()
        
        if booking_type_entry:
            self.booking_type_cache[key] = booking_type_entry.id_bookingType
            return booking_type_entry.id_bookingType
            
        # Create new booking type
        new_booking_type = DimBookingType(
            code=code,
            bookingType=booking_type
        )
        self.session.add(new_booking_type)
        self.session.flush()
        
        self.booking_type_cache[key] = new_booking_type.id_bookingType
        return new_booking_type.id_bookingType
        
    def get_or_create_division_id(self, division):
        """Get or create division dimension entry"""
        # Handle None/NaN
        if pd.isna(division) or division is None or division == '':
            return None
            
        division_lower = division.lower()
        
        if division_lower in self.division_cache:
            return self.division_cache[division_lower]
            
        # Check if division exists
        division_entry = self.session.query(DimDivision).filter_by(
            divisionName=division
        ).first()
        
        if division_entry:
            self.division_cache[division_lower] = division_entry.id_division
            return division_entry.id_division
            
        # Create new division
        new_division = DimDivision(
            divisionName=division
        )
        self.session.add(new_division)
        self.session.flush()
        
        self.division_cache[division_lower] = new_division.id_division
        return new_division.id_division
        
    def booking_exists(self, date_id, time_start_id, time_end_id, room_id, user_id):
        """Check if a booking already exists to avoid duplicates"""
        return self.session.query(FactBookings).filter(
            FactBookings.id_date == date_id,
            FactBookings.id_time_start == time_start_id,
            FactBookings.id_time_end == time_end_id,
            FactBookings.id_room == room_id,
            FactBookings.id_user == user_id
        ).count() > 0
        
    def parse_french_date(self, date_str):
        """Parse French format dates like '8 janv. 2023'"""
        # French month name mapping
        french_months = {
            'janv.': '01', 'févr.': '02', 'mars': '03', 'avr.': '04',
            'mai': '05', 'juin': '06', 'juil.': '07', 'août': '08',
            'sept.': '09', 'oct.': '10', 'nov.': '11', 'déc.': '12'
        }
        
        # Handle various date formats
        try:
            # Split the date string
            parts = date_str.split()
            day = parts[0].zfill(2)  # Ensure day is two digits
            month = None
            year = None
            
            # Find month and year
            for part in parts[1:]:
                if part in french_months:
                    month = french_months[part]
                elif part.isdigit() and len(part) == 4:
                    year = part
                
            if day and month and year:
                # Create datetime object
                return datetime(int(year), int(month), int(day))
            else:
                # Try alternative parsing if components are missing
                return pd.to_datetime(date_str, format='mixed', dayfirst=True)
                
        except Exception as e:
            # Fall back to pandas date parsing
            try:
                return pd.to_datetime(date_str, format='mixed', dayfirst=True)
            except:
                logger.warning(f"Could not parse date: {date_str}, error: {str(e)}")
                # If all parsing fails, return today's date
                return datetime.now()
        
    def process_room_allocations(self):
        """Process data from the RoomAllocations files"""
        try:
            # Check if data directory exists
            if not os.path.exists(BOOKING_DATA_DIR):
                logger.warning(f"BellevueBooking directory not found: {BOOKING_DATA_DIR}")
                return
                
            # Find all room allocation files
            allocation_files = glob.glob(os.path.join(BOOKING_DATA_DIR, "RoomAllocations_*.csv"))
            
            if not allocation_files:
                logger.warning("No RoomAllocations files found!")
                return
                
            batch_count = 0
            total_records = 0
            batch_size = 500  # Commit every 500 records
            
            for file_path in allocation_files:
                logger.info(f"Processing {file_path}")
                
                try:
                    # Read with encoding detection to handle special characters
                    df = pd.read_csv(file_path, encoding='utf-8')
                    
                    # Normalize column names - handle common naming variations
                    column_mapping = {
                        'Nom': 'room_name',
                        'Nom entier': 'room_full_name',
                        'Date': 'booking_date',
                        'Date de dÃ©but': 'start_time',
                        'Date de début': 'start_time',
                        'Date de fin': 'end_time',
                        'Type de rÃ©servation': 'booking_type',
                        'Type de réservation': 'booking_type',
                        'Codes': 'code',
                        "Nom de l'utilisateur": 'username',
                        'Classe': 'class_name',
                        'ActivitÃ©': 'activity',
                        'Activité': 'activity',
                        'Professeur': 'professor',
                        'Division': 'division'
                    }
                    
                    # Rename columns based on mapping
                    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
                    
                    # Process each row
                    for _, row in df.iterrows():
                        try:
                            # Parse date - convert from French format
                            booking_date = self.parse_french_date(row['booking_date'])
                            
                            # Get dimension IDs
                            date_id = self.get_or_create_date_id(booking_date)
                            time_start_id = self.get_or_create_time_id(str(row['start_time']))
                            time_end_id = self.get_or_create_time_id(str(row['end_time']))
                            room_id = self.get_or_create_room_id(row['room_name'], row['room_full_name'])
                            user_id = self.get_or_create_user_id(row['username'])
                            
                            # Skip if missing critical IDs
                            if None in [date_id, time_start_id, time_end_id, room_id, user_id]:
                                logger.warning(f"Skipping row with missing critical data: {row.to_dict()}")
                                continue
                                
                            # Get optional dimension IDs
                            professor_id = self.get_or_create_user_id(row['professor'] if 'professor' in row else None)
                            activity_id = self.get_or_create_activity_id(row['activity'] if 'activity' in row else None)
                            division_id = self.get_or_create_division_id(row['division'] if 'division' in row else None)
                            booking_type_id = self.get_or_create_booking_type_id(
                                row['code'] if 'code' in row else None,
                                row['booking_type'] if 'booking_type' in row else None
                            )
                            
                            # Skip if booking already exists
                            if self.booking_exists(date_id, time_start_id, time_end_id, room_id, user_id):
                                continue
                                
                            # Create new fact record
                            new_fact = FactBookings(
                                id_date=date_id,
                                id_time_start=time_start_id,
                                id_time_end=time_end_id,
                                id_room=room_id,
                                id_user=user_id,
                                id_professor=professor_id,
                                id_activity=activity_id,
                                id_division=division_id,
                                id_bookingType=booking_type_id
                            )
                            self.session.add(new_fact)
                            
                            batch_count += 1
                            total_records += 1
                            
                            # Commit in batches
                            if batch_count >= batch_size:
                                self.session.commit()
                                logger.info(f"Committed batch of {batch_count} records, total: {total_records}")
                                batch_count = 0
                        except Exception as row_error:
                            logger.error(f"Error processing row: {str(row_error)}")
                            logger.debug(f"Problem row: {row.to_dict()}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.session.rollback()
                    continue
                    
            # Commit any remaining records
            if batch_count > 0:
                self.session.commit()
                logger.info(f"Committed final batch of {batch_count} records, total: {total_records}")
                
            logger.info(f"Completed processing Room Allocations data. Total records: {total_records}")
            
        except Exception as e:
            logger.error(f"Error in process_room_allocations: {str(e)}")
            logger.error(traceback.format_exc())
            self.session.rollback()
            raise
            
    def run(self):
        """Run the complete ETL process"""
        try:
            logger.info("Starting Room Allocation ETL process")
            
            # Process room allocation data
            self.process_room_allocations()
            
            logger.info("Room Allocation ETL process completed successfully")
            
        except Exception as e:
            logger.error(f"ETL process failed: {str(e)}")
            logger.error(traceback.format_exc())
            self.session.rollback()
        finally:
            self.session.close()


if __name__ == "__main__":
    etl = RoomAllocationETL()
    etl.run()
