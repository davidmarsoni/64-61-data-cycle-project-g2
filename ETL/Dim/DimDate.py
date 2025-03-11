from sqlalchemy.orm import Session
from ETL.db.models import DimDate
import logging

_date_cache = {}

def get_or_create_date(session: Session, date_obj):
    """
    Add or get date from DimDate
    
    Args:
        session: SQLAlchemy session
        date_obj: datetime object
        
    Returns:
        id_date: Integer representing the date ID
    """
    
    cache_key = (date_obj.year, date_obj.month, date_obj.day)
    
    # Check if the date ID is already in the cache
    if cache_key in _date_cache:
        return _date_cache[cache_key]
    
    # Check if date exists in database
    date_record = session.query(DimDate).filter(
        DimDate.year == date_obj.year,
        DimDate.month == date_obj.month,
        DimDate.day == date_obj.day
    ).first()
    
    if date_record:
        # Cache the result before returning
        _date_cache[cache_key] = date_record.id_date
        return date_record.id_date
    
    # Create new date record
    new_date = DimDate(
        year=date_obj.year,
        month=date_obj.month,
        day=date_obj.day
    )
    
    session.add(new_date)
    session.flush()
    
    _date_cache[cache_key] = new_date.id_date
    
    logging.info(f"Created new date record: Year={new_date.year}, Month={new_date.month}, Day={new_date.day}, ID={new_date.id_date}")
    
    return new_date.id_date

def generate_date_range(session: Session, start_date, end_date):
    """
    Generate date entries for a range of dates
    
    Args:
        session: SQLAlchemy session
        start_date: datetime object representing start date
        end_date: datetime object representing end date
        
    Returns:
        dict: Dictionary mapping (year, month, day) to date IDs
    """
    date_ids = {}
    current_date = start_date
    
    from datetime import timedelta
    while current_date <= end_date:
        date_id = get_or_create_date(session, current_date)
        date_ids[(current_date.year, current_date.month, current_date.day)] = date_id
        current_date += timedelta(days=1)
    
    # Commit changes to ensure all date entries are saved
    session.commit()
    
    logging.info(f"Generated {len(date_ids)} date entries from {start_date.date()} to {end_date.date()}")
    return date_ids
