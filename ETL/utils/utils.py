import pyodbc
from config import ensure_installed

# Ensure required packages are installed
ensure_installed('pyodbc')

def create_connection():
    """Create database connection using Windows Authentication"""
    server = '.' 
    database = 'data_cycle_db'
    try:
        conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
        conn = pyodbc.connect(conn_str)
        print("Connection successful!")
        return conn
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def get_connection_string():
    """
    Get a SQLAlchemy connection string for the same database
    """
    # Using the same server and database as create_connection
    server = '.'
    database = 'data_cycle_db'
    return f"mssql+pyodbc://{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
