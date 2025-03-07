"""
Functions to clean BellevueBooking data files
"""
import os
import logging
import pandas as pd
import csv
import traceback

# Import utilities from the utils module
from clean.utils import log_error

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
