from sqlalchemy.orm import Session
from ETL.db.models import DimInverter

_inverter_cache = {}

def get_or_create_inverter(session: Session, inverter_name):
    """
    Add or get inverter from DimInverter
    
    Args:
        session: SQLAlchemy session
        inverter_name: String name of the inverter
        
    Returns:
        id_inverter: Integer representing the inverter ID
    """
    
    # Check cache first
    if inverter_name in _inverter_cache:
        return _inverter_cache[inverter_name]
    
    # Check if inverter exists
    inverter_record = session.query(DimInverter).filter(
        DimInverter.inverter_name == inverter_name
    ).first()
    
    if inverter_record:
        # Update cache and return
        _inverter_cache[inverter_name] = inverter_record.id_inverter
        return inverter_record.id_inverter
    
    # Create new inverter record
    new_inverter = DimInverter(inverter_name=inverter_name)
    
    session.add(new_inverter)
    session.flush()
    
    # Update cache with new entry
    _inverter_cache[inverter_name] = new_inverter.id_inverter
    
    return new_inverter.id_inverter
