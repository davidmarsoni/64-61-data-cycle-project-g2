from sqlalchemy.orm import Session
from ETL.db.models import DimSite

_site_cache = {}

def get_or_create_site(session: Session, site_name):
    """
    Add or get site from DimSite
    
    Args:
        session: SQLAlchemy session
        site_name: String name of the site
        
    Returns:
        id_site: Integer representing the site ID
    """
    # Check cache first
    if site_name in _site_cache:
        return _site_cache[site_name]
    
    # Check if site exists
    site_record = session.query(DimSite).filter(
        DimSite.site_name == site_name
    ).first()
    
    if site_record:
        # Update cache and return
        _site_cache[site_name] = site_record.id_site
        return site_record.id_site
    
    # Create new site record
    new_site = DimSite(site_name=site_name)
    
    session.add(new_site)
    session.flush()
    
    # Update cache with new entry
    _site_cache[site_name] = new_site.id_site
    
    return new_site.id_site
