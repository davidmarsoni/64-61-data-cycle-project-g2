import pyodbc
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

def create_connection():
    # Connection parameters
    server = 'ZEPHYRUS-ANDREI'  # e.g., 'localhost' or 'DESKTOP-ABC\SQLEXPRESS'
    database = 'data_cycle_db'
   
    
    try:
        # For Windows Authentication, use this instead:
        conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
        
        conn = pyodbc.connect(conn_str)
        print("Connection successful!")
        return conn
    
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def execute_query(connection, query):
    try:
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return rows
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    connection = create_connection()
    if connection:
        # Your database operations here
        connection.close()
        
# Load the file
BASE_DIR = os.getenv('BASE_DIR')
current_date = datetime.now().strftime('%Y-%m-%d')
CLEAN_DATA_DIR = os.path.join(BASE_DIR, f"cleaned_data_{current_date}") if BASE_DIR else None

SUBFOLDERS = {
    "BellevueConso": ["Consumption", "Temperature", "Humidity"]
}

print(CLEAN_DATA_DIR)
print(SUBFOLDERS)

# Get all the files from the folder BellevueConso and get only the file with the above subfolders
def get_files_by_category():
    files_by_category = {}
    
    if not CLEAN_DATA_DIR:
        print("BASE_DIR environment variable not set or CLEAN_DATA_DIR could not be created")
        return files_by_category
    
    for main_folder, categories in SUBFOLDERS.items():
        main_folder_path = os.path.join(CLEAN_DATA_DIR, main_folder)
        if not os.path.exists(main_folder_path):
            print(f"Main folder path does not exist: {main_folder_path}")
            continue
            
        files_by_category[main_folder] = {}
        
        # Get all files in the main folder
        all_files = [f for f in os.listdir(main_folder_path) if os.path.isfile(os.path.join(main_folder_path, f))]
        
        # Organize files by category
        for category in categories:
            # Check if file contains the category name (case-insensitive)
            matching_files = [
                os.path.join(main_folder_path, filename) 
                for filename in all_files 
                if category.lower() in filename.lower()
            ]
            
            # Store the files for this category
            files_by_category[main_folder][category] = matching_files
            
    return files_by_category

# Get organized files
organized_files = get_files_by_category()
print(organized_files)

import pandas as pd
import os
from datetime import datetime
import pyodbc
import numpy as np


def populate_dim_tables_and_facts():
    connection = create_connection()
    if not connection:
        print("Failed to connect to database")
        return
    
    try:
        # Get organized files
        organized_files = get_files_by_category()
        
        # Dictionary to hold dataframes for each category
        dataframes = {}
        all_timestamps = set()
        
        # Read files into dataframes and collect all unique timestamps
        for main_folder, categories in organized_files.items():
            dataframes[main_folder] = {}
            
            for category, file_paths in categories.items():
                if not file_paths:
                    print(f"No {category} files found")
                    continue
                
                # Read the first file for each category (assumes similar structure)
                file_path = file_paths[0]
                print(f"Reading {category} data from {file_path}")
                
                try:
                    # Read CSV file
                    df = pd.read_csv(file_path)
                    print(df.head())
                    
                    # Check if we have separate Date and Time columns or a single Date column
                    if 'Date' in df.columns and 'Time' in df.columns:
                        # If we have separate columns, parse them accordingly
                        # Don't parse_dates yet since we're handling date and time separately
                        print("Found separate Date and Time columns")
                        dataframes[main_folder][category] = df
                        
                        # Collect all unique dates and times
                        all_dates = set(df['Date']) if 'all_dates' in locals() else set()
                        all_times = set(df['Time']) if 'all_times' in locals() else set()
                        
                    elif 'Date' in df.columns:
                        # If we have a single Date column that contains both date and time
                        df['Date'] = pd.to_datetime(df['Date'])
                        dataframes[main_folder][category] = df
                        
                        # Extract date and time parts separately
                        if 'all_dates' not in locals():
                            all_dates = set()
                            all_times = set()
                        
                        # Add to our sets of unique dates and times
                        all_dates.update(df['Date'].dt.strftime('%Y-%m-%d'))
                        all_times.update(df['Date'].dt.strftime('%H:%M:%S'))
                    
                except Exception as e:
                    print(f"Error reading {category} file: {str(e)}")
        
         # Convert dates and times to proper datetime objects for database insertion
        date_objects = {datetime.strptime(date_str, '%Y-%m-%d') for date_str in all_dates}
        time_objects = {datetime.strptime(time_str, '%H:%M:%S') for time_str in all_times}
        
        print("All unique dates:", all_dates)
        print("All unique times:", all_times)
        
        # Populate DimDate table
        date_id_map = {}
        for date_obj in date_objects:
            date_id = populate_dim_date(connection, date_obj)
            date_str = date_obj.strftime('%Y-%m-%d')
            date_id_map[date_str] = date_id
        
        # Populate DimTime table
        time_id_map = {}
        for time_obj in time_objects:
            time_id = populate_dim_time(connection, time_obj)
            time_str = time_obj.strftime('%H:%M:%S')
            time_id_map[time_str] = time_id
    
        
        # Populate FactWeather table
        if 'BellevueConso' in dataframes:
            populate_fact_weather(
                connection, 
                dataframes['BellevueConso'].get('Temperature'), 
                dataframes['BellevueConso'].get('Humidity'),
                dataframes['BellevueConso'].get('Consumption'),
                date_id_map,
                time_id_map
            )
        
        connection.commit()
        print("ETL process completed successfully")
        
    except Exception as e:
        print(f"Error in ETL process: {str(e)}")
        connection.rollback()
    
    finally:
        connection.close()

def populate_dim_date(connection, date_obj):
    """Populate DimDate table and return the date_id"""
    cursor = connection.cursor()
    
    # Check if date already exists
    query = """
    SELECT id_date FROM DimDate 
    WHERE [Year] = ? AND [Month] = ? AND [Day] = ?
    """
    
    print("191 ",date_obj.year, date_obj.month, date_obj.day)
    cursor.execute(query, (date_obj.year, date_obj.month, date_obj.day))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Insert new date
    query = """
    INSERT INTO DimDate ([Year], [Month], [Day])
    VALUES (?, ?, ?)
    """
    cursor.execute(query, (date_obj.year, date_obj.month, date_obj.day))
    
    # Get the new date_id
    cursor.execute("SELECT @@IDENTITY")
    date_id = cursor.fetchone()[0]
    
    return date_id

def populate_dim_time(connection, time_obj):
    """Populate DimTime table and return the time_id"""
    cursor = connection.cursor()
    
    # Check if time already exists
    query = """
    SELECT id_time FROM DimTime 
    WHERE [Hour] = ? AND [Minute] = ?
    """
    cursor.execute(query, (time_obj.hour, time_obj.minute))
    result = cursor.fetchone()
    
    if result:
        return result[0]
    
    # Insert new time
    query = """
    INSERT INTO DimTime ([Hour], [Minute])
    VALUES (?, ?)
    """
    cursor.execute(query, (time_obj.hour, time_obj.minute))
    
    # Get the new time_id
    cursor.execute("SELECT @@IDENTITY")
    time_id = cursor.fetchone()[0]
    
    return time_id

def populate_fact_weather(connection, temp_df, humidity_df, consumption_df, date_id_map, time_id_map):
    """Populate FactWeather table with combined data"""
    cursor = connection.cursor()
    
    # Prepare full dataset from all sources
    all_data = []
    
    # Process all rows in temperature data
    if temp_df is not None:
        for _, row in temp_df.iterrows():
            # Handle either separate Date/Time columns or combined Date column
            if 'Date' in temp_df.columns and 'Time' in temp_df.columns:
                date_str = row['Date']
                time_str = row['Time']
            else:
                # Assuming Date column is already parsed as datetime
                timestamp = row['Date']
                date_str = timestamp.strftime('%Y-%m-%d')
                time_str = timestamp.strftime('%H:%M:%S')
            
            # Skip if we don't have the date or time in our maps
            if date_str not in date_id_map or time_str not in time_id_map:
                continue
            
            # Get the corresponding values from other dataframes
            humidity_value = get_value_for_date_time(humidity_df, date_str, time_str, 'Humidity') if humidity_df is not None else None
            solar_radiation = get_value_for_date_time(consumption_df, date_str, time_str, 'SolarRadiation') if consumption_df is not None else None
            
            # Add data point
            all_data.append({
                'date_id': date_id_map[date_str],
                'time_id': time_id_map[time_str],
                'temperature': row.get('Temperature', None),
                'humidity': humidity_value,
                'solar_radiation': solar_radiation
            })
    
    # Insert data into FactWeather
    for data_point in all_data:
        query = """
        INSERT INTO FactWeather (id_date, id_time, temperature, humidity, solar_radiation)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            data_point['date_id'],
            data_point['time_id'],
            data_point['temperature'],
            data_point['humidity'],
            data_point['solar_radiation']
        ))

def get_value_for_date_time(df, date_str, time_str, column_name):
    """Get value from dataframe for specific date and time and column"""
    if df is None:
        return None
    
    # Handle either separate Date/Time columns or combined Date column
    if 'Date' in df.columns and 'Time' in df.columns:
        # Find matching date and time in dataframe
        matches = df[(df['Date'] == date_str) & (df['Time'] == time_str)]
    else:
        # Find matching timestamp in dataframe with datetime column
        # Convert timestamp strings back to datetime for comparison
        timestamp_str = f"{date_str} {time_str}"
        matches = df[df['Date'] == pd.to_datetime(timestamp_str)]
    
    if not matches.empty and column_name in matches.columns:
        value = matches[column_name].iloc[0]
        return None if pd.isna(value) else value
    
    return None

def get_value_for_timestamp(df, timestamp, column_name):
    """Get value from dataframe for specific timestamp and column"""
    if df is None:
        return None
    
    # Find matching timestamp in dataframe
    matches = df[df['Date'] == timestamp]
    if not matches.empty and column_name in matches.columns:
        value = matches[column_name].iloc[0]
        return None if pd.isna(value) else value
    
    return None

# Add this call at the end of your script
if __name__ == "__main__":
    connection = create_connection()
    if connection:
        # Run ETL process
        populate_dim_tables_and_facts()
        connection.close()

