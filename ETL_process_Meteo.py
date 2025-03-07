# Required library imports
import pyodbc
import os
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables and set configuration
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
current_date = datetime.now().strftime('%Y-%m-%d')
CLEAN_DATA_DIR = os.path.join(BASE_DIR, f"cleaned_data_{current_date}") if BASE_DIR else None

SUBFOLDERS = {
    "Meteo": ["Pred"]
}

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

def get_files_by_category():
    """Get all CSV files organized by category"""
    files_by_category = {}
    
    if not CLEAN_DATA_DIR or not os.path.exists(CLEAN_DATA_DIR):
        print(f"Directory does not exist: {CLEAN_DATA_DIR}")
        return files_by_category
        
    data_folders = [f for f in os.listdir(BASE_DIR) if f.startswith('cleaned_data_')]
    if not data_folders:
        print("No cleaned_data folders found")
        return files_by_category
        
    latest_data_folder = sorted(data_folders)[-1]
    actual_data_dir = os.path.join(BASE_DIR, latest_data_folder)
    
    print(f"Using data folder: {actual_data_dir}")
    
    # Process Meteo folder
    meteo_folder_path = os.path.join(actual_data_dir, "Meteo")
    if not os.path.exists(meteo_folder_path):
        print(f"Meteo folder path does not exist: {meteo_folder_path}")
        return files_by_category
        
    files_by_category["Meteo"] = {}
    
    # Get all Pred CSV files
    for f in os.listdir(meteo_folder_path):
        if f.endswith('.csv') and f.startswith('Pred_'):
            if "Pred" not in files_by_category["Meteo"]:
                files_by_category["Meteo"]["Pred"] = []
            files_by_category["Meteo"]["Pred"].append(os.path.join(meteo_folder_path, f))
            print(f"Found prediction file: {f}")
    
    return files_by_category

def populate_dim_date(connection, date_obj):
    """Add or get date from DimDate"""
    cursor = connection.cursor()
    try:
        # Check if date exists
        cursor.execute("""
            SELECT id_date FROM DimDate 
            WHERE [year] = ? AND [month] = ? AND [day] = ?
        """, (date_obj.year, date_obj.month, date_obj.day))
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Insert new date
        cursor.execute("""
            INSERT INTO DimDate ([year], [month], [day])
            VALUES (?, ?, ?)
        """, (date_obj.year, date_obj.month, date_obj.day))
        connection.commit()
        cursor.execute("SELECT @@IDENTITY")
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error in populate_dim_date: {str(e)}")
        connection.rollback()
        raise

def populate_dim_time(connection, time_obj):
    """Add or get time from DimTime"""
    cursor = connection.cursor()
    try:
        # Check if time exists
        cursor.execute("""
            SELECT id_time FROM DimTime 
            WHERE [hour] = ? AND [minute] = ?
        """, (time_obj.hour, time_obj.minute))
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Insert new time
        cursor.execute("""
            INSERT INTO DimTime ([hour], [minute])
            VALUES (?, ?)
        """, (time_obj.hour, time_obj.minute))
        connection.commit()
        cursor.execute("SELECT @@IDENTITY")
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error in populate_dim_time: {str(e)}")
        connection.rollback()
        raise

def populate_dim_site(connection, site_name):
    """Add or get site from DimSite"""
    cursor = connection.cursor()
    try:
        # Check if site exists
        cursor.execute("SELECT id_site FROM DimSite WHERE siteName = ?", (site_name,))
        result = cursor.fetchone()
        if result:
            return result[0]
        
        # Insert new site
        cursor.execute("INSERT INTO DimSite (siteName) VALUES (?)", (site_name,))
        connection.commit()
        cursor.execute("SELECT @@IDENTITY")
        return cursor.fetchone()[0]
    except Exception as e:
        print(f"Error in populate_dim_site: {str(e)}")
        connection.rollback()
        raise

def get_value_for_date_time(df, date_str, time_str, column_name):
    """Get value from dataframe for specific date and time"""
    if df is None:
        return None
    
    matches = df[(df['Date'] == date_str) & (df['Time'] == time_str)]
    return matches[column_name].iloc[0] if not matches.empty and column_name in matches.columns else None

def validate_meteo_row(row):
    """Validate a single row of meteorological data"""
    try:
        # Check if all required columns exist
        required_columns = ['Date', 'Time', 'Site', 'Prediction', 
                          'PRED_T_2M_ctrl', 'PRED_RELHUM_2M_ctrl', 
                          'PRED_TOT_PREC_ctrl', 'PRED_GLOB_ctrl']
        if not all(col in row.index for col in required_columns):
            return False, "Missing required columns"
        
        # Validate prediction number
        if not isinstance(row['Prediction'], (int, float)) or pd.isna(row['Prediction']):
            return False, "Invalid prediction number"
        
        # Validate measurements
        measurements = {
            'temperature': row['PRED_T_2M_ctrl'],
            'humidity': row['PRED_RELHUM_2M_ctrl'],
            'rain': row['PRED_TOT_PREC_ctrl'],
            'radiation': row['PRED_GLOB_ctrl']
        }
        
        if any(pd.isna(value) for value in measurements.values()):
            return False, "Missing measurement values"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def process_meteo_files(connection, meteo_folder):
    """Process meteorological data files and insert into database"""
    try:
        pred_files = [f for f in os.listdir(meteo_folder) if f.startswith('Pred_') and f.endswith('.csv')]
        stats = {'processed': 0, 'inserted': 0, 'duplicates': 0}
        
        print(f"\nFound {len(pred_files)} files to process")
        
        for i, pred_file in enumerate(pred_files, 1):
            print(f"\nProcessing file {i}/{len(pred_files)}: {pred_file}")
            file_path = os.path.join(meteo_folder, pred_file)
            meteo_df = pd.read_csv(file_path)
            file_stats = {'processed': 0, 'inserted': 0, 'duplicates': 0}
            
            # Standardize dates and times
            meteo_df['Date'] = pd.to_datetime(meteo_df['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')
            meteo_df['Time'] = pd.to_datetime(meteo_df['Time'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
            
            # Create dimension mappings
            mappings = {
                'date': {d: populate_dim_date(connection, datetime.strptime(d, '%Y-%m-%d')) 
                        for d in set(meteo_df['Date'])},
                'time': {t: populate_dim_time(connection, datetime.strptime(t, '%H:%M:%S')) 
                        for t in set(meteo_df['Time'])},
                'site': {s: populate_dim_site(connection, s) 
                        for s in set(meteo_df['Site'])}
            }
            
            cursor = connection.cursor()
            
            # Process each row
            for _, row in meteo_df.iterrows():
                file_stats['processed'] += 1
                
                # Get dimension IDs
                date_id = mappings['date'].get(row['Date'])
                time_id = mappings['time'].get(row['Time'])
                site_id = mappings['site'].get(row['Site'])
                prediction_num = int(row['Prediction'])
                
                # Check for exact duplicate
                cursor.execute("""
                    SELECT 1 FROM FactMeteoSwissData 
                    WHERE id_date = ? AND id_time = ? AND id_site = ? AND numPrediction = ?
                    AND temperature = ? AND humidity = ? AND rain = ? AND radiation = ?
                """, (
                    date_id, time_id, site_id, prediction_num,
                    float(row['PRED_T_2M_ctrl']),
                    float(row['PRED_RELHUM_2M_ctrl']),
                    float(row['PRED_TOT_PREC_ctrl']),
                    float(row['PRED_GLOB_ctrl'])
                ))
                
                if cursor.fetchone():
                    file_stats['duplicates'] += 1
                    continue
                
                # Insert new record
                cursor.execute("""
                    INSERT INTO FactMeteoSwissData 
                    (id_date, id_time, id_site, numPrediction, 
                     temperature, humidity, rain, radiation)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    date_id, time_id, site_id, prediction_num,
                    float(row['PRED_T_2M_ctrl']),
                    float(row['PRED_RELHUM_2M_ctrl']),
                    float(row['PRED_TOT_PREC_ctrl']),
                    float(row['PRED_GLOB_ctrl'])
                ))
                file_stats['inserted'] += 1
                
                # Commit every 10000 rows
                if file_stats['processed'] % 10000 == 0:
                    connection.commit()
            
            connection.commit()
            
            # Update file statistics
            print(f"File complete: {pred_file}")
            print(f"- Rows processed: {file_stats['processed']}")
            print(f"- Rows inserted: {file_stats['inserted']}")
            print(f"- Duplicates found: {file_stats['duplicates']}")
            
            # Update total statistics
            stats['processed'] += file_stats['processed']
            stats['inserted'] += file_stats['inserted']
            stats['duplicates'] += file_stats['duplicates']
            
        print("\n=== Final Summary ===")
        print(f"Total Processed: {stats['processed']}")
        print(f"Total Inserted: {stats['inserted']}")
        print(f"Total Duplicates: {stats['duplicates']}")
        print("===================")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        connection.rollback()
        raise

def populate_dim_tables_and_facts():
    """Main ETL process"""
    connection = create_connection()
    if not connection:
        return
    
    try:
        latest_folder = sorted([f for f in os.listdir(BASE_DIR) if f.startswith('cleaned_data_')])[-1]
        meteo_folder = os.path.join(BASE_DIR, latest_folder, "Meteo")
        
        if os.path.exists(meteo_folder):
            process_meteo_files(connection, meteo_folder)
        else:
            print(f"Error: No Meteo folder in {latest_folder}")
            
    except Exception as e:
        print(f"ETL Error: {str(e)}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    populate_dim_tables_and_facts()
