from sqlalchemy.orm import Session
from ETL.db.models import DimTime
import logging

_time_cache = {}

def get_or_create_time(session: Session, time_obj):
    """
    Add or get time from DimTime
    
    Args:
        session: SQLAlchemy session
        time_obj: datetime object
        
    Returns:
        id_time: Integer representing the time ID
    """
    cache_key = (time_obj.hour, time_obj.minute)
    
    # Check if the time ID is already in the cache
    if cache_key in _time_cache:
        return _time_cache[cache_key]
    
    # Check if time exists in database
    time_record = session.query(DimTime).filter(
        DimTime.hour == time_obj.hour,
        DimTime.minute == time_obj.minute
    ).first()
    
    if time_record:
        # Cache the result before returning
        _time_cache[cache_key] = time_record.id_time
        return time_record.id_time
    
    # Create new time record
    new_time = DimTime(
        hour=time_obj.hour,
        minute=time_obj.minute
    )
    
    session.add(new_time)
    
    # Commit to ensure the record is saved before returning
    session.flush()
    
    # Cache the new ID
    _time_cache[cache_key] = new_time.id_time
    
    logging.info(f"Created new time record: Hour={new_time.hour}, Minute={new_time.minute}, ID={new_time.id_time}")
    
    return new_time.id_time

def generate_time_entries_15min(session: Session):
    """
    Generate time entries at 15-minute intervals (00, 15, 30, 45)
    This is useful to pre-populate the time dimension table
    
    Args:
        session: SQLAlchemy session
        
    Returns:
        dict: Dictionary mapping (hour, minute) to time IDs
    """
    time_ids = {}
    # Generate entries for all hours (0-23) and minutes (0, 15, 30, 45)
    for hour in range(24):
        for minute in [0, 15, 30, 45]:
            from datetime import datetime
            time_obj = datetime(2000, 1, 1, hour, minute)
            
            # Use get_or_create_time to create or get the time entry
            time_id = get_or_create_time(session, time_obj)
            time_ids[(hour, minute)] = time_id
    
    # Commit changes to ensure all time entries are saved
    session.commit()
    
    logging.info(f"Generated {len(time_ids)} time entries at 15-minute intervals")
    return time_ids
