"""
Functions to clean Solarlogs data files
"""
import os
import logging
import pandas as pd
import csv
import traceback

# Import utilities from the utils module instead of data_cleaning
from clean.utils import log_error, detect_encoding

def process_solarlogs(file_path, output_path, encoding, filename):
    """Process Solarlogs files"""
    try:
        # Read the CSV file with the detected encoding
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=';',
            quoting=csv.QUOTE_ALL,
            engine='python'
        )
        
        # Handle the drop of te Unit� affichage colum 3 with the id of the column because the name is not always the same
        data = data.drop(columns=[data.columns[2]], errors="ignore")
        
        # Drop the row "Valeur Acquisition" if it exists
        data = data.drop(columns=['Valeur Acquisition'], errors="ignore")
        
        # Save the filtered data to the output path
        data.to_csv(output_path, index=False)
        logging.info(f"Successfully processed Solarlogs data: {os.path.basename(file_path)}")
        
        return True
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error(f"Process Solarlogs", f"Error processing {filename}: {str(e)}", error_trace)
        return False


def process_min_solarlogs(file_path, output_path):
    """Process min files from Solarlogs with specific transformation"""
    try:
        # First detect encoding
        encoding = detect_encoding(file_path)
        
        logging.info(f"Processing min Solarlogs file: {file_path}")
        
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=';',
            quoting=csv.QUOTE_NONE,
            engine='python'
        )
        
        # Get the first occurrence of Date and Time
        date_time_cols = [col for col in data.columns if 'Date' in col or 'Time' in col][:2]
        date_time_df = data[date_time_cols].copy()
        
        # Get all measurement columns (excluding Date and Time columns)
        measurement_cols = [col for col in data.columns if col not in date_time_cols]
        
        # Create chunks of 11 columns (number of measurements per block)
        chunk_size = 11
        chunks = [measurement_cols[i:i + chunk_size] for i in range(0, len(measurement_cols), chunk_size)]
        
        # Process each chunk and collect rows
        rows = []
        for idx in range(len(data)):
            current_date = date_time_df.iloc[idx, 0]
            current_time = date_time_df.iloc[idx, 1]
            
            # Process each chunk of measurements
            for chunk in chunks:
                row_data = [current_date, current_time]
                row_data.extend(data.loc[idx, chunk].values)
                rows.append(row_data)
        
        # Create new DataFrame with processed data
        columns = ['Date', 'Time', 'INV', 'Pac', 'DaySum', 'Status', 'Error', 'Pdc1', 'Pdc2', 'Udc1', 'Udc2', 'Temp', 'Uac']
        
        result_df = pd.DataFrame(rows, columns=columns)
        
        # Save the cleaned data
        result_df.to_csv(output_path, index=False)
        logging.info(f"Successfully processed min Solarlogs data: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error(f"Process Min Solarlogs", f"Error processing {os.path.basename(file_path)}: {str(e)}", error_trace)
        return False
