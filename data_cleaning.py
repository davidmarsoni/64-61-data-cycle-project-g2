import os
import logging
import pandas as pd
import chardet
import csv
from functools import lru_cache
from pathlib import Path
import concurrent.futures
from datetime import datetime
import traceback

# Import from config file
from config import ensure_installed, Config

# Ensure required packages are installed
ensure_installed('pandas')
ensure_installed('chardet')
ensure_installed('python-dotenv')

def setup_logging():
    # Use Config's setup_logging method instead
    Config.setup_logging('data_cleaning')


@lru_cache(maxsize=100)
def detect_encoding(file_path, num_bytes=10000):
    """Detect file encoding with caching for performance"""
    with open(file_path, 'rb') as f:
        rawdata = f.read(num_bytes)
    result = chardet.detect(rawdata)
    return result['encoding']


def process_solarlogs(file_path, output_path, encoding, filename):
    """Process Solarlogs files"""
    try:
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=';',
            quoting=csv.QUOTE_ALL,
            engine='python'
        )
        
        # Drop unwanted columns (if present)
        data = data.drop(columns=['Unité affichage', 'Valeur Acquisition'], errors="ignore")
        data.to_csv(output_path, index=False)
        logging.info(f"Successfully processed Solarlogs data: {os.path.basename(file_path)}")
        
        return True
    except Exception as e:
        logging.error(f"Error processing Solarlogs file {filename}: {str(e)}")
        logging.error(f"Skipping file {filename}...")
        return False


def process_bellevue_booking(file_path, output_path, encoding, filename):
    """Process BellevueBooking files"""
    try:
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter='\t',
            quoting=csv.QUOTE_NONE,
            engine='python'
        )
        
        # Check if quotes need to be stripped
        if data.iloc[0, 0].startswith('"') and data.iloc[0, -1].endswith('"'):
            # Fix deprecated applymap by using DataFrame.map with a Series
            for col in data.columns:
                if data[col].dtype == 'object':  # Only process string columns
                    data[col] = data[col].map(lambda x: x.strip('"') if isinstance(x, str) else x)
            
            # Clean column headers
            data.columns = [col.strip('"') if isinstance(col, str) else col for col in data.columns]
        
        # Filter by room name that starts with "VS-BEL"
        data = data[data['Nom'].str.startswith('VS-BEL')]
        
        # Drop unwanted columns - modify this based on actual columns
        columns_to_drop = ['Rés.-no', 'Sigle de la salle remplacée', 'Nom entier de la salle remplacée',
                        'Date de début.1', 'Date de fin.1', 'Périodicité', 'Poste de dépenses',
                        'Remarque', 'Annotation']
        data = data.drop(columns=columns_to_drop, errors="ignore")
        
        # Check if 'Classe' and 'Professeur' columns exist
        class_column = 'Classe' if 'Classe' in data.columns else None
        professor_column = 'Professeur' if 'Professeur' in data.columns else None
        
        if not class_column or not professor_column:
            logging.warning(f"Missing required columns in {filename}. Saving filtered data without expansion.")
            data.to_csv(output_path, index=False)
            return True
        
        # Create a new DataFrame to store expanded rows
        
        expanded_rows = []
        
        # Process each row more efficiently
        for _, row in data.iterrows():
            # Split both class and professor fields using vectorized operations
            classes = [cls.strip() for cls in str(row[class_column]).split(',') if cls.strip()]
            professors = [prof.strip() for prof in str(row[professor_column]).split(',') if prof.strip()]
            
            # Skip if either list is empty
            if not classes or not professors:
                continue
                
            # Create a new row for each class and professor combination
            for class_name in classes:
                for professor in professors:
                    new_row = row.copy()
                    new_row[class_column] = class_name
                    new_row[professor_column] = professor
                    expanded_rows.append(new_row)
        
        # Create new DataFrame with expanded rows if any exist
        if expanded_rows:
            result_df = pd.DataFrame(expanded_rows)
            result_df.to_csv(output_path, index=False)
            logging.info(f"Processed {len(data)} original rows into {len(result_df)} expanded rows for {filename}")
        else:
            # Save original filtered data if no expansion happened
            data.to_csv(output_path, index=False)
            logging.info(f"No rows to expand in {filename}. Saved filtered data.")
        
        return True
    except Exception as e:
        logging.error(f"Error processing BellevueBooking file {filename}: {str(e)}")
        logging.error(f"Skipping file {filename}...")
        return False


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
        
        # Use pivot_table to efficiently reshape the data
        df_result = data.pivot_table(
            index=['Prediction', 'Time'],
            columns='Measurement',
            values='Value',
            aggfunc='first'
        ).reset_index()
        
        # Process datetime information - use more efficient method
        df_result['Date'] = pd.to_datetime(df_result['Time']).dt.date
        df_result['Time'] = pd.to_datetime(df_result['Time']).dt.time
        
        # Rearrange columns 
        measurement_cols = [col for col in df_result.columns if col not in ['Prediction', 'Time', 'Date']]
        df_result = df_result[['Date', 'Time', 'Prediction'] + measurement_cols]
        
        # Sort and save
        df_result.sort_values(by=['Prediction', 'Time'], inplace=True)
        df_result.to_csv(output_path, index=False)
        
        logging.info(f"Successfully processed Meteo data: {os.path.basename(file_path)}")
        return True
        
    except Exception as e:
        logging.error(f"Error processing Meteo file {filename}: {str(e)}")
        logging.error(f"Skipping file {filename}...")
        return False
        

def is_file_for_category(filename, category):
    """Check if a file belongs to a specific category based on its prefixes"""
    return any(prefix in filename for prefix in Config.SUBFOLDERS[category])


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
        logging.error(f"Error processing min Solarlogs file: {str(e)}")
        return False


def custom_clean_csv(file_path, output_path):
    """Clean CSV files based on their type and prefix"""
    filename = os.path.basename(file_path)
    logging.info(f"Processing: {filename}")
    
    try:
        # Handle min prefix Solarlogs files differently as they are the only file with a diffeant format
        if is_file_for_category(filename, "Solarlogs") and filename.lower().startswith(Config.MIN_PREFIX_SOLARLOGS):
            return process_min_solarlogs(file_path, output_path)
        
        # For all other files, detect encoding first because some file have a differant encoding
        encoding = detect_encoding(file_path)
        
        # Process files according to their category
        if is_file_for_category(filename, "Solarlogs") or is_file_for_category(filename, "BellevueConso"):
            return process_solarlogs(file_path, output_path, encoding, filename)
            
        elif is_file_for_category(filename, "BellevueBooking"):
            return process_bellevue_booking(file_path, output_path, encoding, filename)
            
        elif is_file_for_category(filename, "Meteo"):
            return process_meteo(file_path, output_path, encoding, filename)
            
        else:
            # Default behavior: copy file as-is TODO need to see if we will no just skip the file
            pd.read_csv(file_path, encoding=encoding).to_csv(output_path, index=False)
            return True
    
    except Exception as e:
        logging.error(f"Error processing {filename}: {str(e)}")
        return False


def process_file(args):
    """Process a single file (for parallel execution)"""
    in_file, out_file = args
    return custom_clean_csv(in_file, out_file)


def process_and_clean_folder(input_folder, output_folder):
    """Process all CSV files in a folder and its subfolders"""
    file_pairs = []
    
    # Collect all files to process
    for root, _, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        if rel_path == ".":
            rel_path = ""
            
        target_folder = Path(output_folder) / rel_path
        target_folder.mkdir(parents=True, exist_ok=True)
        
        csv_files = [f for f in files if f.endswith(".csv")]
        for file in csv_files:
            in_file = Path(root) / file
            out_file = target_folder / file
            file_pairs.append((str(in_file), str(out_file)))
    
    # Optimize max_workers based on system and workload
    max_workers = min(32, os.cpu_count() + 4) if os.cpu_count() else 4
    
    # Process files in parallel with optimized batch size
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_file, file_pairs))
    
    processed = sum(1 for r in results if r)
    logging.info(f"Processed {processed}/{len(file_pairs)} files in {input_folder}")


def main():
    try:
        Config.setup('data_cleaner')
        logging.info(f"Starting data cleaning from {Config.DATA_DIR}")
        logging.info(f"Cleaned files will be stored in: {Config.CLEAN_DATA_DIR}")

        # Process each subfolder individually
        for subfolder in Config.SUBFOLDERS:
            input_folder = Path(Config.DATA_DIR) / subfolder
            output_folder = Path(Config.CLEAN_DATA_DIR) / subfolder
            
            if input_folder.exists():
                logging.info(f"Processing subfolder: {subfolder}")
                process_and_clean_folder(input_folder, output_folder)
            else:
                logging.warning(f"Subfolder {subfolder} does not exist. Skipping.")

        logging.info("Data cleaning completed.")
        
    except Exception as e:
        error_msg = f"Fatal error in data cleaning: {e}"
        error_traceback = traceback.format_exc()
        logging.error(error_msg)
        logging.debug(error_traceback)
        
        # Send email notification about the error using the Config class method
        Config.send_error_email(
            module_name="Data Cleaning",
            subject="Data Cleaning Failure",
            error_message=str(e),
            traceback_info=error_traceback
        )
        raise


if __name__ == "__main__":
    main()