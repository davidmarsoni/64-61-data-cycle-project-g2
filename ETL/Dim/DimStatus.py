from sqlalchemy.orm import Session
from ETL.db.models import DimStatus

_status_cache = {}

def get_or_create_status(session: Session, status_name):
    """
    Add or get status from DimStatus
    
    Args:
        session: SQLAlchemy session
        status_name: String name of the status
        
    Returns:
        id_status: Integer representing the status ID
    """
    # Check cache first
    if status_name in _status_cache:
        return _status_cache[status_name]
    
    # Check if status exists
    status_record = session.query(DimStatus).filter(
        DimStatus.statusName == status_name
    ).first()
    
    if status_record:
        # Update cache and return
        _status_cache[status_name] = status_record.id_status
        return status_record.id_status
    
    # Create new status record
    new_status = DimStatus(statusName=status_name)
    
    session.add(new_status)
    session.flush()
    
    # Update cache with new entry
    _status_cache[status_name] = new_status.id_status
    
    return new_status.id_status