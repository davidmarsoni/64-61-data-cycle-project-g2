import os
import logging
import pandas as pd
from datetime import datetime
import chardet
import csv

# Setup logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

# New date-based input and output directories
CURRENT_DATE = datetime.now().strftime('%Y-%m-%d')
DATA_DIR = rf"C:\DataCollection\collected_data_{CURRENT_DATE}"
CLEAN_DATA_DIR = rf"C:\DataCollection\cleaned_data_{CURRENT_DATE}"

# Dictionary of subfolders and prefixes for custom cleaning
SUBFOLDERS = {
    "Solarlogs": ["PV", "min"],
    "BellevueConso": ["Consumption", "Temperature", "Humidity"],
    "BellevueBooking": ["RoomAllocations"],
    "Meteo": ["Pred"]
}

# Define a constant for the Solarlogs min prefix
MIN_PREFIX_SOLARLOGS = SUBFOLDERS["Solarlogs"][1]

def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        rawdata = f.read(num_bytes)
    result = chardet.detect(rawdata)
    return result['encoding']

# Custom cleaning function based on file prefix
def custom_clean_csv(file_path, output_path):
    logging.info(f"Custom cleaning {file_path} -> {output_path}")
    filename = os.path.basename(file_path)
    encoding = detect_encoding(file_path)

    # If the file belongs to Solarlogs and starts with the designated min prefix
    if any(prefix in filename for prefix in SUBFOLDERS["Solarlogs"]) and filename.lower().startswith(MIN_PREFIX_SOLARLOGS):
        logging.info(f"Skipping file {filename} due to {MIN_PREFIX_SOLARLOGS} prefix, TODO need to clean it")
        return

    # Process Solarlogs (excluding min files) and BellevueConso files.
    if any(prefix in filename for prefix in SUBFOLDERS["Solarlogs"]) or \
       any(prefix in filename for prefix in SUBFOLDERS["BellevueConso"]):

        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=';',
            quoting=csv.QUOTE_ALL,
            engine='python',
            on_bad_lines='skip'
        )
        
        # Drop unwanted columns (if present)
        data = data.drop(columns=['Unit� affichage', 'Valeur Acquisition'], errors="ignore")

        data.to_csv(output_path, index=False)

    elif any(prefix in filename for prefix in SUBFOLDERS["BellevueBooking"]):
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter='\t',
            quoting=csv.QUOTE_ALL,
            engine='python',
            on_bad_lines='skip'
        )
        # Filter by room name that starts with "VS-BEL"
        data = data[data['Nom'].str.startswith('VS-BEL')]
        data.to_csv(output_path, index=False)
        
    elif any(prefix in filename for prefix in SUBFOLDERS["Meteo"]):
        # Filter the 'Site' column for Sion and Visp
        print(file_path)
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=',',
            quoting=csv.QUOTE_NONE,
            engine='python',
            on_bad_lines='skip'
        )
        
        # Filter the 'Site' column for Sion and Visp
        data = data[data['Site'].isin(['Sion', 'Visp'])]

        # Replace -99999.0 with 0
        data.replace(-99999.0, 0, inplace=True)

        # List all unique measurements
        measurements = data['Measurement'].unique()

        # Create a list to store new rows
        result_rows = []

        # For each unique prediction and time, get the measurement values and create a new row
        for prediction in data['Prediction'].unique():
            for time in data['Time'].unique():
                # For each site
                for site in data['Site'].unique():
                    measurement_values = []  # reset for each time and site
                    # Skip if no data for this combination
                    if len(data[(data['Time'] == time) & (data['Site'] == site) & (data['Prediction'] == prediction)]) == 0:
                        continue
                        
                    for measurement in measurements:
                        # Get the value for the current time, site and measurement
                        try:
                            value = data[(data['Time'] == time) & 
                                            (data['Site'] == site) & 
                                            (data['Prediction'] == prediction) & 
                                            (data['Measurement'] == measurement)]['Value'].values[0]
                        except IndexError:
                            # Handle case when measurement doesn't exist for this site/time
                            value = 0
                        measurement_values.append(value)

                    # Create a new row with the prediction, time, site and all measurements
                    result_row = [prediction, time, site] + measurement_values
                    result_rows.append(result_row)
            
        # Create a new DataFrame with the new rows
        df_result = pd.DataFrame(result_rows, columns=['Prediction', 'Time', 'Site'] + list(measurements))

        # Split the Time column into Date and Time
        df_result['Date'] = pd.to_datetime(df_result['Time']).dt.date
        df_result['Time'] = pd.to_datetime(df_result['Time']).dt.time

        # Reorder the columns with Date and Time first
        df_result = df_result[['Date', 'Time', 'Prediction', 'Site'] + list(measurements)]

        # Order the columns by Prediction, Time and all measured values
        df_result = df_result.sort_values(by=['Prediction', 'Time'])
        
        # Save the cleaned data to a new CSV file
        df_result.to_csv(output_path, index=False)
    
    else:
        # Default cleaning
        pass

# Recursively process and clean CSV files, preserving subfolder structure.
def process_and_clean_folder(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        # Determine relative subfolder path
        rel_path = os.path.relpath(root, input_folder)
        if rel_path == ".":
            rel_path = ""  # Remove the dot from the path
        target_folder = os.path.join(output_folder, rel_path)
        os.makedirs(target_folder, exist_ok=True)
        for file in files:
            if file.endswith(".csv"):
                in_file = os.path.join(root, file)
                out_file = os.path.join(target_folder, file)
                print(in_file)
                custom_clean_csv(in_file, out_file)

def main():
    setup_logging()
    logging.info(f"Starting data cleaning in {DATA_DIR}")
    
    if not os.path.exists(DATA_DIR):
        logging.error(f"Data directory {DATA_DIR} does not exist.")
        return

    os.makedirs(CLEAN_DATA_DIR, exist_ok=True)
    logging.info(f"Cleaned files will be stored in: {CLEAN_DATA_DIR}")

    # Process each subfolder individually
    for subfolder in SUBFOLDERS:
        input_folder = os.path.join(DATA_DIR, subfolder)
        output_folder = os.path.join(CLEAN_DATA_DIR, subfolder)
        if os.path.exists(input_folder):
            process_and_clean_folder(input_folder, output_folder)
        else:
            logging.warning(f"Subfolder {input_folder} does not exist. Skipping.")

    logging.info("Data cleaning completed.")

if __name__ == "__main__":
    main()
