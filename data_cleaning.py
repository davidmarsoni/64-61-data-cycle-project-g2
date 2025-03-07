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
import sys

# Import from config file
from config import ensure_installed, Config

# Ensure required packages are installed
ensure_installed('pandas')
ensure_installed('chardet')
ensure_installed('python-dotenv')

# Track errors throughout the execution
error_log = []

def log_error(phase, error_msg, trace=None):
    """Log an error message with optional traceback.

    Args:
        phase (_type_): The phase of execution where the error occurred
        error_msg (_type_): The error message
        trace (_type_, optional): Traceback information. Defaults to None.
    """
    error_entry = {
        'phase': phase,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'message': str(error_msg),
        'traceback': trace
    }
    error_log.append(error_entry)
    if trace:
        logging.error(f"{phase}: {error_msg}")
        logging.debug(trace)
    else:
        logging.error(f"{phase}: {error_msg}")

def send_error_summary():
    """Send an email summarizing all errors encountered during execution
    """
    if not error_log:
        return
    
    # Prepare error message for email
    summary_msg = "The following errors occurred during data cleaning:\n\n"
    for error in error_log:
        summary_msg += f"- Phase: {error['phase']}\n"
        summary_msg += f"  Time: {error['time']}\n"
        summary_msg += f"  Message: {error['message']}\n\n"
    
    # Get the first error's traceback for detailed information
    first_traceback = error_log[0].get('traceback', None) if error_log else None
    
    try:
        Config.send_error_email(
            module_name="Data Cleaner",
            subject=f"Data Cleaning Errors ({len(error_log)} issues)",
            error_message=summary_msg,
            traceback_info=first_traceback
        )
        logging.info("Error summary email sent successfully.")
    except Exception as email_error:
        logging.error(f"Failed to send error summary email: {email_error}")

def load_processed_files(record_file):
    """Load previously processed files from a record file.

    Args:
        record_file (_type_): path to the record file

    Returns:
        _type_: set of processed files
    """
    try:
        if os.path.exists(record_file):
            with open(record_file, "r") as f:
                return set(line.strip() for line in f)
        return set()
    except Exception as e:
        log_error("Load Processed Files", f"Error loading records from {record_file}: {e}", traceback.format_exc())
        return set()  # Return empty set to continue operation

def save_processed_files(record_file, processed):
    """Save the list of processed files to a record file.

    Args:
        record_file (_type_): path to the record file
        processed (_type_): set of processed files
    """
    try:
        with open(record_file, "w") as f:
            for item in processed:
                f.write(f"{item}\n")
    except Exception as e:
        log_error("Save Processed Files", f"Error saving records to {record_file}: {e}", traceback.format_exc())


@lru_cache(maxsize=100)
def detect_encoding(file_path, num_bytes=10000):
    """Detect the encoding of a file using chardet.

    Args:
        file_path (_type_): path to the file ethier relative or absolute
        num_bytes (int, optional): number of bytes to read for detection. Defaults to 10000.

    Returns:
        _type_: detected encoding
    """
    with open(file_path, 'rb') as f:
        rawdata = f.read(num_bytes)
    result = chardet.detect(rawdata)
    return result['encoding']


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
         # Clean column headers first
        data.columns = [col.strip('"') for col in data.columns]
        
        # remove trailing and ending quotes from for all the line if they are present
        data = data.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
        
        print(data.head())
        
        # correct the date format from 8 janv. 2023 to 2023-01-08 
        month_mapping = {
            'janv.': '01', 'févr.': '02', 'mars': '03', 'avr.': '04',
            'mai': '05', 'juin': '06', 'juil.': '07', 'août': '08',
            'sept.': '09', 'oct.': '10', 'nov.': '11', 'déc.': '12'
        }

        def custom_parse_date(date_str):
            if pd.isna(date_str):
                return None
            
            parts = date_str.split()
            if len(parts) != 3:
                return None
                
            day, month_abbr, year = parts
            if month_abbr not in month_mapping:
                return None
                
            month = month_mapping[month_abbr]
            return f"{year}-{month}-{day.zfill(2)}"

        # Save the original date for logging purposes
        data['Date_original'] = data['Date']
        data['Date'] = data['Date'].apply(custom_parse_date)

        # Try converting to datetime and mark invalid dates as NaT
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')

        # Print the row with the invalid date
        invalid_dates = data[data['Date'].isnull()]
        if not invalid_dates.empty:
            logging.warning(f"Invalid dates found in {filename}: {len(invalid_dates)} rows")
            for index, row in invalid_dates.iterrows():
                logging.warning(
                    f"Row {index}: Original Date: {row['Date_original']} - Parsed Date: {row['Date']} - {row['Nom']} - {row['Date de début']} - {row['Date de fin']}"
                )

        # Drop rows with invalid dates
        data = data.dropna(subset=['Date'])
        # TODO =============================
                
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
        columns_to_drop = ['Nom entier','Rés.-no', 'Sigle de la salle remplacée', 'Nom entier de la salle remplacée',
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
        
        # Store the renamed column names
        class_column_renamed = 'class'
        professor_column_renamed = 'professor'
        
        # Rename Nom,Nom entier,Type de réservation,Codes,Nom de l'utilisateur,Classe,Activité,Professeur,Division
        data.rename(columns={'Date': 'Date', 'Date de début': 'start_time', 'Date de fin': 'end_time',
                            'Nom': 'room_name', 'Type de réservation': 'reservation_type',
                            'Codes': 'codes', 'Nom de l\'utilisateur': 'user_name', 
                            'Classe': 'class', 'Activité': 'activity', 
                            'Professeur': 'professor', 'Division': 'division'}, inplace=True)
        
        data = data[['Date', 'start_time', 'end_time'] + [col for col in data.columns if col not in ['Date', 'start_time', 'end_time']]]
        
        # Process each row more efficiently
        for _, row in data.iterrows():
            # Split both class and professor fields using vectorized operations
            classes = [cls.strip() for cls in str(row[class_column_renamed]).split(',') if cls.strip()]
            professors = [prof.strip() for prof in str(row[professor_column_renamed]).split(',') if prof.strip()]
            
            # Skip if either list is empty
            if not classes or not professors:
                continue
                
            # Create a new row for each class and professor combination
            for class_name in classes:
                for professor in professors:
                    new_row = row.copy()
                    new_row[class_column_renamed] = class_name
                    new_row[professor_column_renamed] = professor
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
        error_trace = traceback.format_exc()
        log_error(f"Process BellevueBooking", f"Error processing {filename}: {str(e)}", error_trace)
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
        error_trace = traceback.format_exc()
        log_error(f"Process Min Solarlogs", f"Error processing {os.path.basename(file_path)}: {str(e)}", error_trace)
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
        error_trace = traceback.format_exc()
        log_error(f"Custom Clean CSV", f"Error processing {filename}: {str(e)}", error_trace)
        return False


def process_file(args):
    """Process a single file (for parallel execution)"""
    in_file, out_file, processed_files = args
    filename = os.path.basename(in_file)
    
    # Skip if already processed
    if filename in processed_files:
        logging.info(f"File already processed, skipping: {filename}")
        return True
    
    result = custom_clean_csv(in_file, out_file)
    if result:
        # Add to processed files only if successful
        processed_files.add(filename)
    return result


def process_and_clean_folder(input_folder, output_folder, processed_files, record_file):
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
            file_pairs.append((str(in_file), str(out_file), processed_files))
    
    # Optimize max_workers based on system and workload
    max_workers = min(32, os.cpu_count() + 4) if os.cpu_count() else 4
    
    # Process files in parallel with optimized batch size
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_file, file_pairs))
    
    # Save processed files record
    save_processed_files(record_file, processed_files)
    
    processed = sum(1 for r in results if r)
    logging.info(f"Processed {processed}/{len(file_pairs)} files in {input_folder}")


def main():
    critical_error = False
    try:
        Config.setup('data_cleaner')
        logging.info(f"Starting data cleaning from {Config.DATA_DIR}")
        logging.info(f"Cleaned files will be stored in: {Config.CLEAN_DATA_DIR}")

        # Ensure the record file directory exists
        record_file = os.path.join(Config.BASE_DIR, 'cleaned_files.txt')
        os.makedirs(os.path.dirname(record_file), exist_ok=True)
        
        # Load the set of already processed files
        processed_files = load_processed_files(record_file)
        logging.info(f"Loaded {len(processed_files)} previously processed files")

        # Process each subfolder individually
        for subfolder in Config.SUBFOLDERS:
            input_folder = Path(Config.DATA_DIR) / subfolder
            output_folder = Path(Config.CLEAN_DATA_DIR) / subfolder
            
            if input_folder.exists():
                logging.info(f"Processing subfolder: {subfolder}")
                process_and_clean_folder(input_folder, output_folder, processed_files, record_file)
            else:
                logging.warning(f"Subfolder {subfolder} does not exist. Skipping.")

        logging.info("Data cleaning completed.")
        
    except Exception as e:
        critical_error = True
        error_msg = f"Fatal error in data cleaning: {e}"
        error_traceback = traceback.format_exc()
        log_error("Main Execution", error_msg, error_traceback)
    finally:
        # Send error summary email if there were any errors
        if error_log:
            send_error_summary()
        
        if critical_error:
            sys.exit(1)


if __name__ == "__main__":
    main()