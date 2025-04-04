from ETL.db.models import DimBookingType

def get_or_create_booking_type(session, code, booking_type):
    """Get existing booking type or create new one"""
    booking_type_entry = session.query(DimBookingType).filter_by(
        code=code,
        bookingType=booking_type
    ).first()
    if not booking_type_entry:
        booking_type_entry = DimBookingType(
            code=code,
            bookingType=booking_type
        )
        session.add(booking_type_entry)
        session.commit()
    return booking_type_entry.id_bookingtype