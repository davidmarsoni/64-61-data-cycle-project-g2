"""
Functions to clean Meteo data files
"""
import os
import logging
import pandas as pd
import csv
import traceback

# Import utilities from the utils module
from clean.utils import log_error

def process_meteo(file_path, output_path, encoding, filename):
    """Process Meteo files"""
    try:
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=',',
            quoting=csv.QUOTE_ALL,
            engine='python',
            on_bad_lines='skip'
        )
        
        # Replace -99999.0 with 0
        data.replace(-99999.0, 0, inplace=True)
        
        # Check if Site column exists and filter for only Sion and Visp
        has_site_column = 'Site' in data.columns
        if has_site_column:
            # Filter for only Sion and Visp sites
            data = data[data['Site'].isin(['Sion', 'Visp'])]
            
            # Check if we have data left after filtering
            if data.empty:
                logging.warning(f"No data for Sion or Visp in file {filename}, skipping")
                return False
        
        # Use pivot_table to efficiently reshape the data
        # Include Site in index if it exists
        index_cols = ['Prediction', 'Time']
        if has_site_column:
            index_cols.insert(0, 'Site')
            
        df_result = data.pivot_table(
            index=index_cols,
            columns='Measurement',
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        # Process datetime information - use more efficient method
        df_result['Date'] = pd.to_datetime(df_result['Time']).dt.date
        df_result['Time'] = pd.to_datetime(df_result['Time']).dt.time
        
        # Rearrange columns in the requested order: Date, Time, Site, Prediction, then measurement columns
        first_cols = ['Date', 'Time']
        if has_site_column:
            first_cols.append('Site')
        first_cols.append('Prediction')
            
        measurement_cols = [col for col in df_result.columns 
                          if col not in first_cols + ['Date', 'Time']]
        df_result = df_result[first_cols + measurement_cols]
        
        # Sort using the requested order as well
        sort_cols = ['Date', 'Time']
        if has_site_column:
            sort_cols.append('Site')
        sort_cols.append('Prediction')
            
        df_result.sort_values(by=sort_cols, inplace=True)
        df_result.to_csv(output_path, index=False)
        
        logging.info(f"Successfully processed Meteo data: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        error_trace = traceback.format_exc()
        log_error(f"Process Meteo", f"Error processing {filename}: {str(e)}", error_trace)
        return False
