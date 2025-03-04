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

def execute_query(connection, query):
    """Execute a SQL query and return results"""
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        print(f"Error executing query: {str(e)}")
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
    
    query = """
    SELECT id_date FROM DimDate 
    WHERE [Year] = ? AND [Month] = ? AND [Day] = ?
    """
    
    try:
        cursor.execute(query, (date_obj.year, date_obj.month, date_obj.day))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        insert_query = """
        INSERT INTO DimDate ([Year], [Month], [Day])
        VALUES (?, ?, ?)
        """
        cursor.execute(insert_query, (date_obj.year, date_obj.month, date_obj.day))
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
    
    query = """
    SELECT id_time FROM DimTime 
    WHERE [Hour] = ? AND [Minute] = ?
    """
    
    try:
        cursor.execute(query, (time_obj.hour, time_obj.minute))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        insert_query = """
        INSERT INTO DimTime ([Hour], [Minute])
        VALUES (?, ?)
        """
        cursor.execute(insert_query, (time_obj.hour, time_obj.minute))
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
    if not matches.empty and column_name in matches.columns:
        value = matches[column_name].iloc[0]
        return None if pd.isna(value) else value
    
    return None

def populate_fact_weather(connection, temp_df, humidity_df, consumption_df, date_id_map, time_id_map):
    """Populate FactWeather table with checks for existing records"""
    cursor = connection.cursor()
    
    try:
        # Process temperature data with corresponding humidity
        if temp_df is not None:
            for _, row in temp_df.iterrows():
                date_str = row['Date']
                time_str = row['Time']
                
                # Get IDs from dimension tables
                date_id = date_id_map.get(date_str)
                time_id = time_id_map.get(time_str)
                
                if date_id and time_id:
                    # Check if record already exists
                    check_query = """
                    SELECT 1 FROM FactWeather 
                    WHERE id_date = ? AND id_time = ?
                    """
                    cursor.execute(check_query, (date_id, time_id))
                    if cursor.fetchone():
                        continue
                    
                    # Get humidity value for same date/time
                    humidity_value = get_value_for_date_time(humidity_df, date_str, time_str, 'Variation')
                    
                    # Insert new record
                    insert_query = """
                    INSERT INTO FactWeather (id_date, id_time, temperature, humidity)
                    VALUES (?, ?, ?, ?)
                    """
                    cursor.execute(insert_query, (
                        date_id,
                        time_id,
                        row['Variation'],  # Temperature value
                        humidity_value
                    ))
            
            connection.commit()
            
    except Exception as e:
        print(f"Error in populate_fact_weather: {str(e)}")
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
            for category, file_paths in categories.items():
                if not file_paths:
                    print(f"No {category} files found")
                    continue
                
                for file_path in file_paths:
                    print(f"Reading {category} data from {file_path}")
                    
                    try:
                        df = pd.read_csv(file_path)
                        file_date = os.path.basename(file_path).split('-')[0]
                        
                        df['Date'] = pd.to_datetime(df['Date'], format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
                        df['Time'] = pd.to_datetime(df['Heure'], format='%H:%M:%S').dt.strftime('%H:%M:%S')
                        
                        if category == 'Temperature':
                            humidity_path = next((f for f in organized_files[main_folder]['Humidity'] 
                                               if file_date in f), None)
                            
                            if humidity_path:
                                humidity_df = pd.read_csv(humidity_path)
                                humidity_df['Date'] = pd.to_datetime(humidity_df['Date'], 
                                                                   format='%d.%m.%Y').dt.strftime('%Y-%m-%d')
                                humidity_df['Time'] = pd.to_datetime(humidity_df['Heure'], 
                                                                   format='%H:%M:%S').dt.strftime('%H:%M:%S')
                                
                                all_dates.update(df['Date'])
                                all_times.update(df['Time'])
                                
                                date_id_map = {
                                    date_str: populate_dim_date(connection, datetime.strptime(date_str, '%Y-%m-%d'))
                                    for date_str in all_dates
                                }
                                
                                time_id_map = {
                                    time_str: populate_dim_time(connection, datetime.strptime(time_str, '%H:%M:%S'))
                                    for time_str in all_times
                                }
                                
                                populate_fact_weather(
                                    connection,
                                    df,
                                    humidity_df,
                                    None,
                                    date_id_map,
                                    time_id_map
                                )
                                
                    except Exception as e:
                        print(f"Error processing file {file_path}: {str(e)}")
                        continue
        
        connection.commit()
        print("ETL process completed successfully")
        
    except Exception as e:
        print(f"Error in ETL process: {str(e)}")
        connection.rollback()
    finally:
        connection.close()

if __name__ == "__main__":
    populate_dim_tables_and_facts()
