from ETL.db.models import DimUser

def get_or_create_user(session, user_name):
    """Get existing user or create new one"""
    user = session.query(DimUser).filter_by(username=user_name).first()
    if not user:
        user = DimUser(username=user_name)
        session.add(user)
        session.commit()
    return user.id_user