from sqlalchemy import Column, Integer, String
from ETL.db.models import DimActivity

def get_or_create_activity(session, activity_name):
    """Get existing activity or create new one"""
    activity = session.query(DimActivity).filter_by(activity_name=activity_name).first()
    if not activity:
        activity = DimActivity(activity_name=activity_name)
        session.add(activity)
        session.commit()
    return activity.id_activity