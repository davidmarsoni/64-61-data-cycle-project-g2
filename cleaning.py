import os
import logging
import pandas as pd
import csv
from pathlib import Path
import concurrent.futures
import sys
import traceback

# Import from config file
from config import ensure_installed, Config

# Ensure required packages are installed
ensure_installed('pandas')
ensure_installed('chardet')
ensure_installed('python-dotenv')

# Import utilities from the clean.utils module
from clean.utils import log_error, detect_encoding, error_log

# Import the specialized cleaning functions
from clean.solarlogs import process_solarlogs, process_min_solarlogs
from clean.bellevue_booking import process_bellevue_booking
from clean.meteo import process_meteo

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

def is_file_for_category(filename, category):
    """Check if a file belongs to a specific category based on its prefixes"""
    return any(prefix in filename for prefix in Config.SUBFOLDERS[category])

def custom_clean_csv(file_path, output_path):
    """Clean CSV files based on their type and prefix"""
    filename = os.path.basename(file_path)
    logging.info(f"Processing: {filename}")
    
    try:
        # Handle min prefix Solarlogs files differently as they are the only file with a different format
        if is_file_for_category(filename, "Solarlogs") and filename.lower().startswith(Config.MIN_PREFIX_SOLARLOGS):
            return process_min_solarlogs(file_path, output_path)
        
        # For all other files, detect encoding first because some file have a different encoding
        encoding = detect_encoding(file_path)
        
        # Process files according to their category
        if is_file_for_category(filename, "Solarlogs") or is_file_for_category(filename, "BellevueConso"):
            return process_solarlogs(file_path, output_path, encoding, filename)
            
        elif is_file_for_category(filename, "BellevueBooking"):
            return process_bellevue_booking(file_path, output_path, encoding, filename)
            
        elif is_file_for_category(filename, "Meteo"):
            return process_meteo(file_path, output_path, encoding, filename)
            
        else:
            # Default behavior: copy file as-is
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