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
        
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].apply(lambda x: x.strip('"') if isinstance(x, str) else x)
    
        # Handle different column name formats
        # Map possible date column names to standard 'Date'
        date_column_variants = ['Date', 'date', 'Date de réservation', 'DateReservation']
        found_date_column = None
        for col_name in date_column_variants:
            if col_name in data.columns:
                found_date_column = col_name
                break
                
        if found_date_column is None:
            # If no date column is found, try to construct one from other columns
            if 'Année' in data.columns and 'Mois' in data.columns and 'Jour' in data.columns:
                # Construct date from separate year, month, day columns - always use 2023
                data['Date'] = '2023' + '-' + data['Mois'].astype(str).str.zfill(2) + '-' + data['Jour'].astype(str).str.zfill(2)
                logging.info(f"Created Date column from Mois/Jour columns in {filename} (enforcing year 2023)")
            else:
                # Log the available columns for debugging
                logging.error(f"No date column found in {filename}. Available columns: {', '.join(data.columns)}")
                return False
        else:
            # Standardize the date column name
            data.rename(columns={found_date_column: 'Date'}, inplace=True)
            logging.info(f"Using {found_date_column} as Date column in {filename}")
     
        # Save the original date for logging purposes
        data['Date_original'] = data['Date']
            
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
            # Always use 2023 regardless of the actual year
            return f"2023-{month}-{day.zfill(2)}"
        
        # Check if the dates look like "8 janv. 2023" format
        sample_date = data['Date'].iloc[0] if len(data) > 0 else ""
        if isinstance(sample_date, str) and len(sample_date.split()) == 3:
            data['Date'] = data['Date'].apply(custom_parse_date)
        else:
            # For other date formats, extract month and day but force year to be 2023
            try:
                # First convert to datetime to standardize
                temp_dates = pd.to_datetime(data['Date'], errors='coerce')
                # Then rebuild with 2023 as the year
                data['Date'] = temp_dates.dt.strftime('2023-%m-%d')
                # Count how many dates were changed
                year_changed = sum(temp_dates.dt.year != 2023)
                if year_changed > 0:
                    logging.warning(f"Changed {year_changed} dates to use year 2023 in {filename}")
            except Exception as date_err:
                logging.error(f"Error standardizing dates to 2023 in {filename}: {str(date_err)}")
        
        # Try converting to datetime and mark invalid dates as NaT
        data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d', errors='coerce')
        
        # Print the row with the invalid date
        invalid_dates = data[data['Date'].isnull()]
        if not invalid_dates.empty:
            logging.warning(f"Invalid dates found in {filename}: {len(invalid_dates)} rows")
            # You can uncomment the following lines to log the details of invalid dates
            # for index, row in invalid_dates.iterrows():
            #    logging.warning(
            #        f"Row {index}: Original Date: {row['Date_original']}"
            #    )
        # Drop rows with invalid dates
        data = data.dropna(subset=['Date'])

        # Additional step to ensure all dates use 2023 as the year
        if not data.empty:
            changed_years = sum(data['Date'].dt.year != 2023)
            if changed_years > 0:
                data['Date'] = data['Date'].apply(lambda x: x.replace(year=2023) if not pd.isna(x) else x)
                logging.warning(f"Forced {changed_years} dates to use year 2023 in {filename}")
        
        # Check if quotes need to be stripped
        if len(data) > 0 and isinstance(data.iloc[0, 0], str) and data.iloc[0, 0].startswith('"') and isinstance(data.iloc[0, -1], str) and data.iloc[0, -1].endswith('"'):
            # Use apply instead of deprecated applymap
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].apply(lambda x: x.strip('"') if isinstance(x, str) else x)
            
            # Clean column headers
            data.columns = [col.strip('"') if isinstance(col, str) else col for col in data.columns]
        
        # Map various room name column variations
        room_column_variants = ['Nom', 'nom', 'Salle', 'salle', 'Room']
        found_room_column = None
        for col_name in room_column_variants:
            if col_name in data.columns:
                found_room_column = col_name
                break
                
        if found_room_column is None:
            logging.error(f"No room name column found in {filename}")
            return False
            
        # Standardize the room column name
        data.rename(columns={found_room_column: 'room_name'}, inplace=True)
            
        # Filter by room name that starts with "VS-BEL"
        data = data[data['room_name'].astype(str).str.startswith('VS-BEL')]
        
        # Drop unwanted columns - modify this based on actual columns
        columns_to_drop = ['Nom entier','Rés.-no', 'Sigle de la salle remplacée', 'Nom entier de la salle remplacée',
                        'Date de début.1', 'Date de fin.1', 'Périodicité', 'Poste de dépenses',
                        'Remarque', 'Annotation']
        data = data.drop(columns=columns_to_drop, errors="ignore")
        
        # Map various start/end time column names to standard names
        start_time_variants = ['Date de début', 'Start Time', 'Heure début', 'start_time', 'Start', 'Début']
        end_time_variants = ['Date de fin', 'End Time', 'Heure fin', 'end_time', 'End', 'Fin']
        
        # Find start time column
        found_start_time = None
        for col_name in start_time_variants:
            if col_name in data.columns:
                found_start_time = col_name
                break
                
        # Find end time column
        found_end_time = None
        for col_name in end_time_variants:
            if col_name in data.columns:
                found_end_time = col_name
                break
        
        # Map other common column variations
        column_mapping = {
            'Type de réservation': 'reservation_type',
            'TypeReservation': 'reservation_type',
            'Type': 'reservation_type',
            'Codes': 'codes',
            'Code': 'codes',
            'Nom de l\'utilisateur': 'user_name',
            'Utilisateur': 'user_name',
            'User': 'user_name',
            'Classe': 'class',
            'Class': 'class',
            'Classes': 'class',
            'Activité': 'activity',
            'Activity': 'activity',
            'Professeur': 'professor',
            'Professor': 'professor',
            'Division': 'division',
            'Department': 'division'
        }
        
        # Rename all matching columns to standardized names
        rename_dict = {}
        if found_start_time:
            rename_dict[found_start_time] = 'start_time'
        if found_end_time:
            rename_dict[found_end_time] = 'end_time'
            
        # Add the rest of the column mappings
        for orig_col, std_col in column_mapping.items():
            if orig_col in data.columns:
                rename_dict[orig_col] = std_col
                
        # Apply all the renames at once
        data.rename(columns=rename_dict, inplace=True)
        
        # Check for required columns
        required_columns = ['Date', 'start_time', 'end_time', 'room_name']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            logging.error(f"Missing required columns in {filename}: {', '.join(missing_columns)}")
            return False
        
        # Check if 'class' and 'professor' columns exist (using standardized names now)
        class_column = 'class' if 'class' in data.columns else None
        professor_column = 'professor' if 'professor' in data.columns else None
        
        # Handle missing class or professor columns
        if not class_column or not professor_column:
            logging.warning(f"Missing class or professor columns in {filename}. Saving filtered data without expansion.")
            data.to_csv(output_path, index=False)
            return True
        
        # Create a new DataFrame to store expanded rows
        expanded_rows = []
        
        # Ensure the columns needed for output are in the correct order
        columns_order = ['Date', 'start_time', 'end_time'] + [col for col in data.columns if col not in ['Date', 'start_time', 'end_time']]
        data = data[columns_order]
        
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
        error_trace = traceback.format_exc()
        log_error(f"Process BellevueBooking", f"Error processing {filename}: {str(e)}", error_trace)
        return False