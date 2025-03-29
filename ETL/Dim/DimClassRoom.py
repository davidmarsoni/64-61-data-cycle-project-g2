import logging
from sqlalchemy.exc import SQLAlchemyError
from ETL.db.models import DimClassRoom
from ETL.utils.logging_utils import log_error

def get_or_create_classroom(session, classroom_name):
    """
    Get or create a classroom record in the DimClassRoom table
    
    Args:
        session: SQLAlchemy session
        classroom_name: Name of the classroom
        
    Returns:
        id_classroom: The ID of the classroom
    """
    if not classroom_name or classroom_name.lower() == 'none':
        return None
    
    try:
        # Check if classroom already exists
        classroom = session.query(DimClassRoom).filter(
            DimClassRoom.classroomName == classroom_name
        ).first()
        
        if classroom:
            return classroom.id_classroom
        
        # Create new classroom
        new_classroom = DimClassRoom(classroomName=classroom_name)
        session.add(new_classroom)
        session.flush()  # Flush to get the ID without committing transaction
        
        logging.info(f"Created new classroom '{classroom_name}' with ID {new_classroom.id_classroom}")
        
        return new_classroom.id_classroom
        
    except SQLAlchemyError as e:
        log_error("Database", f"Error creating classroom '{classroom_name}': {str(e)}")
        raise