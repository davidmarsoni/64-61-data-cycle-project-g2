from ETL.db.models import DimRoom

def get_or_create_room(session, room_name):
    """Get existing room or create new one"""
    room = session.query(DimRoom).filter_by(roomName=room_name).first()
    if not room:
        room = DimRoom(roomName=room_name)  # roomFullName removed as it's not useful
        session.add(room)
        session.commit()
    return room.id_room