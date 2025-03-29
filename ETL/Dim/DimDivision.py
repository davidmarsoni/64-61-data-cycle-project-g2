from ETL.db.models import DimDivision

def get_or_create_division(session, division_name):
    """Get existing division or create new one"""
    division = session.query(DimDivision).filter_by(divisionName=division_name).first()
    if not division:
        division = DimDivision(divisionName=division_name)
        session.add(division)
        session.commit()
    return division.id_division