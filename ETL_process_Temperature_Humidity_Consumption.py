# Required library imports
import pyodbc
import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Global configuration
BASE_DIR = os.getenv('BASE_DIR')
current_date = datetime.now().strftime('%Y-%m-%d')
CLEAN_DATA_DIR = os.path.join(BASE_DIR, f"cleaned_data_{current_date}") if BASE_DIR else None

SUBFOLDERS = {
    "BellevueConso": ["Consumption", "Temperature", "Humidity"]
}

def create_connection():
    """Create database connection using Windows Authentication"""
    server = '.' #I used this so that it selects the local server automatically  
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
        
    # List all folders in the cleaned_data directory to find the most recent one
    data_folders = [f for f in os.listdir(BASE_DIR) if f.startswith('cleaned_data_')]
    if not data_folders:
        print("No cleaned_data folders found")
        return files_by_category
        
    # Get the most recent data folder
    latest_data_folder = sorted(data_folders)[-1]
    actual_data_dir = os.path.join(BASE_DIR, latest_data_folder)
    
    print(f"Using data folder: {actual_data_dir}")
    
    # Process BellevueConso folder
    main_folder_path = os.path.join(actual_data_dir, "BellevueConso")
    if not os.path.exists(main_folder_path):
        print(f"Main folder path does not exist: {main_folder_path}")
        return files_by_category
        
    files_by_category["BellevueConso"] = {}
    
    # Get all CSV files and categorize them by type
    for f in os.listdir(main_folder_path):
        if f.endswith('.csv'):
            for category in SUBFOLDERS["BellevueConso"]:
                if category.lower() in f.lower():
                    if category not in files_by_category["BellevueConso"]:
                        files_by_category["BellevueConso"][category] = []
                    files_by_category["BellevueConso"][category].append(os.path.join(main_folder_path, f))
                    print(f"Found {category} file: {f}")
    
    return files_by_category

def populate_dim_date(connection, date_obj):
    """Populate DimDate table and prevent duplicates"""
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
            SELECT id_date FROM DimDate 
            WHERE [year] = ? AND [month] = ? AND [day] = ?
        """, (date_obj.year, date_obj.month, date_obj.day))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
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
    """Populate DimTime table and prevent duplicates"""
    cursor = connection.cursor()
    
    try:
        cursor.execute("""
            SELECT id_time FROM DimTime 
            WHERE [hour] = ? AND [minute] = ?
        """, (time_obj.hour, time_obj.minute))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
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

def get_value_for_date_time(df, date_str, time_str, column_name):
    """Get value from dataframe for specific date and time"""
    if df is None:
        return None
    
    matches = df[(df['Date'] == date_str) & (df['Time'] == time_str)]
    return matches[column_name].iloc[0] if not matches.empty and column_name in matches.columns else None

def populate_fact_energy_consumption(connection, consumption_df, temp_df, humidity_df, date_id_map, time_id_map):
    """Populate FactEnergyConsumption table with energy, temperature and humidity data"""
    cursor = connection.cursor()
    
    try:
        if consumption_df is not None:
            for _, row in consumption_df.iterrows():
                date_str = row['Date']
                time_str = row['Time']
                
                date_id = date_id_map.get(date_str)
                time_id = time_id_map.get(time_str)
                
                if date_id and time_id:
                    cursor.execute("""
                        SELECT 1 FROM FactEnergyConsumption 
                        WHERE id_date = ? AND id_time = ?
                    """, (date_id, time_id))
                    if cursor.fetchone():
                        continue
                    
                    temperature = get_value_for_date_time(temp_df, date_str, time_str, 'Variation')
                    humidity = get_value_for_date_time(humidity_df, date_str, time_str, 'Variation')
                    
                    if temperature is None or humidity is None:
                        continue
                    
                    cursor.execute("""
                        INSERT INTO FactEnergyConsumption (id_date, id_time, energy_consumed, temperature, humidity)
                        VALUES (?, ?, ?, ?, ?)
                    """, (date_id, time_id, row['Variation'], temperature, humidity))
            
            connection.commit()
            
    except Exception as e:
        print(f"Error in populate_fact_energy_consumption: {str(e)}")
        connection.rollback()
        raise

def populate_dim_tables_and_facts():
    """Main ETL process to populate all tables"""
    connection = create_connection()
    if not connection:
        print("Failed to connect to database")
        return
    
    try:
        organized_files = get_files_by_category()
        all_dates = set()
        all_times = set()
        
        for main_folder, categories in organized_files.items():
            # Process files by date to ensure we have all related data
            consumption_files = categories.get('Consumption', [])
            
            for consumption_file in consumption_files:
                try:
                    file_date = os.path.basename(consumption_file).split('-')[0]
                    
                    # Find corresponding temperature and humidity files
                    temp_file = next((f for f in categories.get('Temperature', []) 
                                    if file_date in f), None)
                    humidity_file = next((f for f in categories.get('Humidity', []) 
                                        if file_date in f), None)
                    
                    if not all([temp_file, humidity_file]):
                        print(f"Missing temperature or humidity data for {file_date}")
                        continue
                    
                    # Read and preprocess all files
                    consumption_df = pd.read_csv(consumption_file)
                    temp_df = pd.read_csv(temp_file)
                    humidity_df = pd.read_csv(humidity_file)
                    
                    # Standardize date/time formats
                    for df in [consumption_df, temp_df, humidity_df]:
                        df['Date'] = pd.to_datetime(df['Date'], 
                                                  format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
                        df['Time'] = pd.to_datetime(df['Heure'], 
                                                  format='%H:%M:%S').dt.strftime('%H:%M:%S')
                    
                    # Update dimension mappings
                    all_dates.update(consumption_df['Date'])
                    all_times.update(consumption_df['Time'])
                    
                    date_id_map = {
                        date_str: populate_dim_date(connection, 
                                                  datetime.strptime(date_str, '%Y-%m-%d'))
                        for date_str in all_dates
                    }
                    
                    time_id_map = {
                        time_str: populate_dim_time(connection, 
                                                  datetime.strptime(time_str, '%H:%M:%S'))
                        for time_str in all_times
                    }
                    
                    # Populate fact table
                    populate_fact_energy_consumption(
                        connection,
                        consumption_df,
                        temp_df,
                        humidity_df,
                        date_id_map,
                        time_id_map
                    )
                    
                except Exception as e:
                    print(f"Error processing file {consumption_file}: {str(e)}")
                    continue
        
        print("ETL process completed successfully")
        
    except Exception as e:
        print(f"Error in ETL process: {str(e)}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    populate_dim_tables_and_facts()
