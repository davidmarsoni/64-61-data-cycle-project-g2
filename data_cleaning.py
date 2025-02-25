import os
import io
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
    "Meteo": ["Meteo"]
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
        logging.info(f"Transforming file {filename} from repeated header to single header")

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
        columns = ['Date', 'Time', 'INV','Pac','DaySum','Status','Error','Pdc1','Pdc2','Udc1','Udc2','Temp','Uac']
        # debug count the number of columns
        logging.info(f"Number of columns: {len(columns)}")
        # debug count the number of rows
        logging.info(f"Number of rows: {len(rows)}")
        result_df = pd.DataFrame(rows, columns=columns)
        
        # Save the cleaned data
        result_df.to_csv(output_path, index=False)
        logging.info(f"Processed {len(chunks)} measurement blocks into {len(result_df)} rows")
        return

    # Process Solarlogs (excluding min files) and BellevueConso files.
    if any(prefix in filename for prefix in SUBFOLDERS["Solarlogs"]) or \
       any(prefix in filename for prefix in SUBFOLDERS["BellevueConso"]):

        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=';',
            quoting=csv.QUOTE_ALL,
            engine='python'
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
            engine='python'
        )
        # Filter by room name that starts with "VS-BEL"
        data = data[data['Nom'].str.startswith('VS-BEL')]
        
        # Drop unwanted columns
        columns_to_drop = ['Rés.-no', 'Sigle de la salle remplacée', 'Nom entier de la salle remplacée',
                        'Date de début.1', 'Date de fin.1', 'Périodicité', 'Poste de dépenses',
                        'Remarque', 'Annotation']
        data = data.drop(columns=columns_to_drop, errors="ignore")
        
        # Create a new DataFrame to store expanded rows
        expanded_rows = []
        
        # Process each row
        for _, row in data.iterrows():
            # Split both class and professor fields
            classes = str(row['Classe']).split(',')  # Handle potential NaN values
            professors = str(row['Professeur']).split(',')  # Handle potential NaN values
            
            # Create a new row for each class and professor combination
            for class_name in classes:
                class_name = class_name.strip()
                if not class_name:  # Skip empty classes
                    continue
                    
                for professor in professors:
                    professor = professor.strip()
                    if not professor:  # Skip empty professors
                        continue
                        
                    new_row = row.copy()
                    new_row['Classe'] = class_name
                    new_row['Professeur'] = professor
                    expanded_rows.append(new_row)
        
        # Create new DataFrame with expanded rows
        result_df = pd.DataFrame(expanded_rows)
        
        # Save the cleaned and expanded data
        result_df.to_csv(output_path, index=False)
        logging.info(f"Processed {len(data)} original rows into {len(result_df)} expanded rows")
        return
    elif any(prefix in filename for prefix in SUBFOLDERS["Meteo"]):
        # Filter the 'Site' column for Sion and Visp
        data = pd.read_csv(
            file_path,
            encoding=encoding,
            delimiter=',',
            quoting=csv.QUOTE_NONE,
            engine='python'
        )
        print(data.head())
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
                measurement_values = []  # reset for each time
                for measurement in measurements:
                    # Get the value for the current time and measurement
                    value = data[(data['Time'] == time) & (data['Measurement'] == measurement)]['Value'].values[0]
                    measurement_values.append(value)

                # Create a new row with the prediction, time and all measurements
                result_row = [prediction, time] + measurement_values
                result_rows.append(result_row)
            
        # Create a new DataFrame with the new rows
        df_result = pd.DataFrame(result_rows, columns=['Prediction', 'Time'] + list(measurements))

        # Split the Time column into Date and Time
        df_result['Date'] = pd.to_datetime(df_result['Time']).dt.date
        df_result['Time'] = pd.to_datetime(df_result['Time']).dt.time

        # Reorder the columns with Date and Time first
        df_result = df_result[['Date', 'Time', 'Prediction'] + list(measurements)]

        # Order the columns by Prediction, Time and all measured values
        df_result = df_result.sort_values(by=['Prediction', 'Time'])
        return
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