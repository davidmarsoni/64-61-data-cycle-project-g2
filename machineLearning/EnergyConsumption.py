import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import pickle
import re, glob

# Models
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

# ----------------------------------
# Functions for loading data
# ----------------------------------

import os, glob, re
from datetime import datetime, timedelta

def get_directories(data_date=None, exclude_last_n=3):
    if data_date is None:
        data_date = datetime.today().strftime('%Y-%m-%d')
    ref_dt = datetime.strptime(data_date, '%Y-%m-%d')

    base_dir = os.getenv('BASE_DIR', r'C:\DataCollection')
    pattern  = os.path.join(base_dir, 'cleaned_data_*')
    all_dirs = glob.glob(pattern)

    def extract_dt(p):
        m = re.search(r'cleaned_data_(\d{4}-\d{2}-\d{2})$', p)
        return datetime.strptime(m.group(1), '%Y-%m-%d') if m else None

    # [(path, date)] 
    dated = sorted(
        [(p, extract_dt(p)) for p in all_dirs if extract_dt(p) is not None],
        key=lambda x: x[1]
    )

    # Ne garder que ceux <= (ref_dt - exclude_last_n jours)
    cutoff = ref_dt - timedelta(days=exclude_last_n)
    filtered = [p for p, d in dated if d <= cutoff]

    # Construire les listes de sous-dossiers
    all_conso_dirs = []
    all_meteo_dirs = []
    all_room_dirs  = []
    for d in filtered:
        if os.path.isdir(os.path.join(d, 'BellevueConso')):
            all_conso_dirs.append(os.path.join(d, 'BellevueConso'))
        if os.path.isdir(os.path.join(d, 'Meteo')):
            all_meteo_dirs.append(os.path.join(d, 'Meteo'))
        if os.path.isdir(os.path.join(d, 'BellevueBooking')):
            all_room_dirs.append(os.path.join(d, 'BellevueBooking'))

    specific = os.path.join(base_dir, f'cleaned_data_{data_date}')
    return {
        'base_dir': base_dir,
        'specific_data_dir': specific,
        'all_conso_dirs': all_conso_dirs,
        'all_meteo_dirs': all_meteo_dirs,
        'all_room_dirs':  all_room_dirs,
        'conso_dir':  os.path.join(specific, 'BellevueConso'),
        'meteo_dir':  os.path.join(specific, 'Meteo'),
        'room_dir':   os.path.join(specific, 'BellevueBooking'),
    }


def load_csv_files(directories, file_pattern=None, cutoff_date=None):
    """
    Loads multiple CSV files from multiple directories with optional pattern filtering and date filtering.
    
    Parameters:
        directories (list): List of directory paths to search for CSV files
        file_pattern (str, optional): Pattern to match in filenames
        cutoff_date (datetime, optional): Cutoff date for filtering data
        
    Returns:
        DataFrame: Combined data from all matching CSV files
    """
   
    
    
    all_data = []
    files_processed = 0
    files_skipped = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            continue
            
        print(f"Scanning directory: {directory}")
        
        # Extract date from directory name if possible
        dir_date = None
        date_match = re.search(r'cleaned_data_(\d{4}-\d{2}-\d{2})', directory)
        if date_match:
            try:
                dir_date = datetime.strptime(date_match.group(1), '%Y-%m-%d')
                print(f"  Directory date: {dir_date.date()}")
                
                # Skip directory if it's beyond cutoff date
                if cutoff_date is not None and dir_date > cutoff_date:
                    print(f"  Skipping directory (beyond cutoff date): {directory}")
                    files_skipped += 1
                    continue
            except ValueError:
                pass
        
        for filename in os.listdir(directory):
            if not filename.endswith('.csv'):
                continue
                
            if file_pattern and file_pattern not in filename:
                continue
            
            # Extract date from filename if possible
            file_date = None
            file_date_patterns = [
                r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                r'(\d{4})(\d{2})(\d{2})'  # YYYYMMDD
            ]
            
            for pattern in file_date_patterns:
                match = re.search(pattern, filename)
                if match:
                    try:
                        if len(match.groups()) == 1:
                            file_date = datetime.strptime(match.group(1), '%Y-%m-%d')
                        elif len(match.groups()) == 3:
                            year, month, day = match.groups()
                            file_date = datetime(int(year), int(month), int(day))
                        
                        # Skip file if it's beyond cutoff date
                        if cutoff_date is not None and file_date > cutoff_date:
                            print(f"  Skipping file (beyond cutoff date): {filename}")
                            files_skipped += 1
                            continue
                    except ValueError:
                        pass
                    break
            
            file_path = os.path.join(directory, filename)
            df = None
            
            # Try multiple encodings
            encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    
                    df['source_file'] = filename
                    df['source_dir'] = directory
                    
                    # Apply date cutoff if the file has date/time columns
                    if cutoff_date is not None and 'date' in df.columns and 'time' in df.columns:
                        # Create datetime column
                        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                        
                        # Filter rows by cutoff date
                        original_count = len(df)
                        df = df[df['datetime'] <= cutoff_date]
                        filtered_count = len(df)
                        
                        if filtered_count < original_count:
                            print(f"  Filtered {filename}: removed {original_count - filtered_count} rows after cutoff date")
                        
                        # Skip empty dataframes after filtering
                        if df.empty:
                            print(f"  Skipping {filename} - all data is beyond cutoff date")
                            files_skipped += 1
                            break
                    
                    all_data.append(df)
                    files_processed += 1
                    print(f"  Loaded {filename} with encoding {encoding}, {len(df)} rows")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"  Error loading {filename}: {e}")
                    break
    
    print(f"Files processed: {files_processed}")
    print(f"Files skipped due to date filtering: {files_skipped}")
    
    if not all_data:
        print("No matching files found.")
        return pd.DataFrame()
    
    # Combine dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final cutoff filter on combined data
    if cutoff_date is not None and 'datetime' in combined_df.columns:
        original_count = len(combined_df)
        combined_df = combined_df[combined_df['datetime'] <= cutoff_date]
        filtered_count = len(combined_df)
        
        if filtered_count < original_count:
            print(f"Final filter: removed {original_count - filtered_count} rows beyond cutoff date {cutoff_date}")
    
    return combined_df

def load_consumption_data(directories, cutoff_date=None):
    """
    Prepares energy consumption data from multiple directories.
    
    Parameters:
        directories (list): List of directories to scan for consumption files
        cutoff_date (datetime, optional): Cutoff date for filtering data
        
    Returns:
        DataFrame: Prepared consumption data
    """
    # Load CSV files from all directories
    consumption_df = load_csv_files(directories, file_pattern='Consumption', cutoff_date=cutoff_date)
    
    if consumption_df.empty:
        print("Warning: No consumption data found.")
        return consumption_df
    
    # Create a copy to avoid modifying the original DataFrame
    df = consumption_df.copy()
    
    # Standardize column names (lowercase)
    df.columns = [col.lower() for col in df.columns]
    
    # Verify required columns
    required_cols = ['date', 'time', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Required columns missing in consumption data: {missing_cols}")
        return pd.DataFrame()
    
    # Ensure datetime column exists
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    
    # Remove rows with invalid dates
    invalid_dates = df['datetime'].isna()
    if invalid_dates.any():
        print(f"Warning: Removing {invalid_dates.sum()} rows with invalid datetime")
        df = df[~invalid_dates].copy()
    
    # Final cutoff date filtering
    if cutoff_date is not None:
        original_count = len(df)
        df = df[df['datetime'] <= cutoff_date]
        filtered_count = len(df)
        if filtered_count < original_count:
            print(f"Final consumption data filter: removed {original_count - filtered_count} rows beyond cutoff date")
    
    # Rename 'value' column to 'consumption'
    df = df.rename(columns={'value': 'consumption'})
    
    # Set datetime as index and sort
    df = df.set_index('datetime').sort_index()
    
    # Return only the consumption column
    return df[['consumption']]

def load_prediction_weather_data(directories, cutoff_date=None, select_highest_prediction=True):
    """
    Loads and processes weather prediction data from multiple directories.
    
    Parameters:
        directories (list): List of directories to scan for weather prediction files
        cutoff_date (datetime, optional): Cutoff date for filtering data
        select_highest_prediction (bool): Whether to select the most recent prediction for each hour
        
    Returns:
        DataFrame: Processed weather prediction data
    """
    # Load CSV files from all directories
    pred_files = load_csv_files(directories, file_pattern='Pred_', cutoff_date=cutoff_date)
    
    if pred_files.empty:
        print("Warning: No weather prediction data found.")
        return pred_files
    
    try:
        df = pred_files.copy()
        
        # Display original columns for verification
        original_columns = df.columns.tolist()
        print(f"Original meteo columns: {original_columns}")
        
        # Standardize column names (lowercase)
        df.columns = [col.lower() for col in df.columns]
        
        # Check for required columns with the EXACT structure you have
        column_mapping = {
            'pred_t_2m_ctrl': 'temperature',
            'pred_relhum_2m_ctrl': 'humidity',
            'pred_tot_prec_ctrl': 'precipitation',
            'pred_glob_ctrl': 'global_radiation'
        }
        
        # Verify columns after lowercase conversion
        print(f"Meteo columns after lowercase: {df.columns.tolist()}")
        
        # Perform column mapping
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
                print(f"Renamed column {old_name} to {new_name}")
        
        # Verify date and time columns
        if 'date' in df.columns and 'time' in df.columns:
            print(f"Sample date values: {df['date'].head().tolist()}")
            print(f"Sample time values: {df['time'].head().tolist()}")
            
            # Convert date and time to datetime
            try:
                # First attempt with standard format
                df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
                print(f"First attempt datetime conversion: {df['datetime'].head()}")
            except Exception as e:
                print(f"Error in first datetime conversion attempt: {e}")
                try:
                    # Second attempt with format adjustment
                    df['datetime'] = pd.to_datetime(pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d') + ' ' + df['time'], errors='coerce')
                    print(f"Second attempt datetime conversion: {df['datetime'].head()}")
                except Exception as e:
                    print(f"Error in second datetime conversion attempt: {e}")
                    return pd.DataFrame()
            
            # Remove rows with invalid dates
            invalid_dates = df['datetime'].isna()
            if invalid_dates.any():
                print(f"Warning: Found {invalid_dates.sum()} invalid datetime entries")
                df = df[~invalid_dates].copy()
            
            # Apply cutoff date filtering
            if cutoff_date is not None:
                original_count = len(df)
                df = df[df['datetime'] <= cutoff_date]
                filtered_count = len(df)
                if filtered_count < original_count:
                    print(f"Final weather pred filter: removed {original_count - filtered_count} rows beyond cutoff date")
            
            # Handle prediction column and filter by most recent predictions
            if 'prediction' in df.columns and select_highest_prediction:
                print("Selecting most recent predictions (highest 'prediction' value)...")
                
                total_pred_count = len(df)
                
                idx = df.groupby('datetime')['prediction'].idxmax().tolist()
                df_filtered = df.loc[idx]
                
                if df_filtered.empty:
                    print("Warning: No valid predictions found after grouping")
                    return pd.DataFrame()
                
                df = df_filtered.copy()
                
                filtered_pred_count = len(df)
                print(f"Reduced predictions from {total_pred_count} to {filtered_pred_count} (selecting most recent for each hour)")
            
            # Set datetime as index and sort
            df = df.set_index('datetime').sort_index()
            
            # Check available weather columns
            weather_cols = ['temperature', 'humidity', 'precipitation', 'global_radiation']
            available_cols = [col for col in weather_cols if col in df.columns]
            
            # If no mapped columns found, try with original column names
            if not available_cols:
                print("No weather columns found after mapping, trying original column names...")
                
                # Try to detect weather columns directly
                potential_weather_cols = [col for col in df.columns if 
                                         any(s in col.lower() for s in ['temp', 'hum', 'prec', 'glob', 'ctrl'])]
                
                if potential_weather_cols:
                    print(f"Found potential weather columns: {potential_weather_cols}")
                    
                    # Mapping based on keywords in column names
                    if any('t_2m' in col.lower() or 'temp' in col.lower() for col in potential_weather_cols):
                        temp_col = next(col for col in potential_weather_cols if 't_2m' in col.lower() or 'temp' in col.lower())
                        df['temperature'] = df[temp_col]
                        available_cols.append('temperature')
                    
                    if any('hum' in col.lower() for col in potential_weather_cols):
                        hum_col = next(col for col in potential_weather_cols if 'hum' in col.lower())
                        df['humidity'] = df[hum_col]
                        available_cols.append('humidity')
                    
                    if any('prec' in col.lower() for col in potential_weather_cols):
                        prec_col = next(col for col in potential_weather_cols if 'prec' in col.lower())
                        df['precipitation'] = df[prec_col]
                        available_cols.append('precipitation')
                    
                    if any('glob' in col.lower() for col in potential_weather_cols):
                        glob_col = next(col for col in potential_weather_cols if 'glob' in col.lower())
                        df['global_radiation'] = df[glob_col]
                        available_cols.append('global_radiation')
            
            if not available_cols:
                print("Warning: No relevant weather columns found.")
                return pd.DataFrame()
            
            print(f"Found weather columns: {available_cols}")
            weather_df = df[available_cols].copy()
            
            # Create derived columns
            if 'temperature' in weather_df.columns:
                print(f"Temperature column summary: min={weather_df['temperature'].min()}, max={weather_df['temperature'].max()}")
                weather_df.loc[:, 'heating_degree'] = weather_df['temperature'].apply(lambda x: max(18.0 - x, 0))
                weather_df.loc[:, 'cooling_degree'] = weather_df['temperature'].apply(lambda x: max(x - 22.0, 0))
                
                weather_df.loc[:, 'temp_change_1h'] = weather_df['temperature'].diff()
                
                if len(weather_df) >= 24:
                    weather_df.loc[:, 'date'] = weather_df.index.date
                    
                    daily_min = weather_df.groupby('date')['temperature'].min()
                    daily_max = weather_df.groupby('date')['temperature'].max()
                    
                    daily_amplitude = (daily_max - daily_min).to_dict()
                    
                    weather_df.loc[:, 'daily_temp_amplitude'] = weather_df['date'].map(daily_amplitude)
                    
                    weather_df = weather_df.drop(columns=['date'])
            
            if 'precipitation' in weather_df.columns:
                weather_df.loc[:, 'is_raining'] = (weather_df['precipitation'] > 0).astype(int)
                
                if len(weather_df) >= 24:
                    weather_df.loc[:, 'precipitation_24h'] = weather_df['precipitation'].rolling(window=24, min_periods=1).sum()
            
            print(f"Final weather DataFrame shape: {weather_df.shape}")
            print(f"Weather index sample: {weather_df.index[:5]}")
            return weather_df
            
        else:
            print(f"Warning: No 'date' or 'time' columns found in prediction data. Columns: {df.columns.tolist()}")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error while processing meteo prediction data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_room_allocation_data_hourly(directory):
    """
    Load room allocation data and transform it into an hourly DataFrame
    Adapted for the new CSV format
    """
    try:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            return pd.DataFrame()
        
        room_files = [f for f in os.listdir(directory) if f.endswith('.csv') and 'RoomAllocations' in f]
        
        if not room_files:
            print("No allocation file found.")
            return pd.DataFrame()
        
        room_files.sort(reverse=True) 
        
        latest_file = room_files[0]
        file_path = os.path.join(directory, latest_file)
        
        print(f"Treating allocation file: {latest_file}")
        
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"Loaded {latest_file} with encoding {encoding}, {len(df)} lines")
                break  # Load successful
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error when loading {latest_file}: {e}")
                return pd.DataFrame()
        
        if df is None:
            print(f"Unable to load {latest_file} with available encodings.")
            return pd.DataFrame()
        
        df.columns = [col.lower() for col in df.columns]
        
        print(f"Columns available: {df.columns.tolist()}")
        
        # Check if we have the expected columns for the new format
        if 'date' in df.columns and 'start_time' in df.columns and 'end_time' in df.columns:
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            
            df = df.dropna(subset=['date'])
            
            def parse_time(time_str):
                """Convert time string to decimal hours."""
                try:
                    if ':' in time_str:
                        hours, minutes = map(int, time_str.split(':'))
                        return hours + minutes / 60
                    else:
                        return float(time_str)
                except:
                    return None
            
            df['heure_debut_decimal'] = df['start_time'].apply(parse_time)
            df['heure_fin_decimal'] = df['end_time'].apply(parse_time)
            
            df = df.dropna(subset=['heure_debut_decimal', 'heure_fin_decimal'])
            
            min_date = df['date'].min()
            max_date = df['date'].max()
            
            hourly_index = pd.date_range(
                start=min_date.replace(hour=0, minute=0), 
                end=max_date.replace(hour=23, minute=59), 
                freq='h'  
            )
            
            hourly_df = pd.DataFrame(index=hourly_index)
            hourly_df.index.name = 'datetime'
            
            room_counts = []
            
            # Use room_name instead of nom
            total_rooms = len(df['room_name'].unique())
            
            for dt in hourly_index:
                date = dt.date()
                hour = dt.hour
                
                day_reservations = df[df['date'].dt.date == date]
                
                occupied_count = sum(
                    (day_reservations['heure_debut_decimal'] <= hour) & 
                    (day_reservations['heure_fin_decimal'] > hour)
                )
                
                room_counts.append({
                    'datetime': dt,
                    'rooms_occupied': occupied_count,
                    'total_rooms': total_rooms,
                    'occupation_rate': occupied_count / max(1, total_rooms)  
                })
            
            result_df = pd.DataFrame(room_counts)
            result_df.set_index('datetime', inplace=True)
            
            result_df['hour'] = result_df.index.hour
            result_df['day_of_week'] = result_df.index.dayofweek
            result_df['is_weekend'] = (result_df['day_of_week'] >= 5).astype(int)
            result_df['is_business_hours'] = ((result_df['hour'] >= 8) & 
                                             (result_df['hour'] < 18) & 
                                             (result_df['is_weekend'] == 0)).astype(int)
            
            high_threshold = result_df['occupation_rate'].quantile(0.75)
            low_threshold = result_df['occupation_rate'].quantile(0.25)
            
            result_df['is_high_occupation'] = (result_df['occupation_rate'] > high_threshold).astype(int)
            result_df['is_low_occupation'] = (result_df['occupation_rate'] < low_threshold).astype(int)
            
            # Add more features based on the additional data available
            if 'activity' in df.columns:
                # Create hourly course activity stats
                for dt in hourly_index:
                    date = dt.date()
                    hour = dt.hour
                    
                    hour_reservations = df[
                        (df['date'].dt.date == date) & 
                        (df['heure_debut_decimal'] <= hour) & 
                        (df['heure_fin_decimal'] > hour)
                    ]
                    
                    # Count courses vs other activities
                    courses_count = sum(hour_reservations['activity'].str.lower() == 'cours') if not hour_reservations.empty else 0
                    
                    # Add to the dataframe
                    result_df.loc[dt, 'course_count'] = courses_count
                
                # Calculate course ratio
                result_df['course_ratio'] = result_df['course_count'] / result_df['rooms_occupied'].replace(0, 1)
                result_df['is_course_heavy'] = (result_df['course_ratio'] > 0.7).astype(int)
            
            # Add division statistics if available
            if 'division' in df.columns:
                divisions = df['division'].dropna().unique()
                
                for division in divisions:
                    if not isinstance(division, str):
                        continue
                        
                    division_name = division.lower().replace(' ', '_').replace('é', 'e').replace('è', 'e').replace('à', 'a')
                    
                    # Create hourly division counts
                    for dt in hourly_index:
                        date = dt.date()
                        hour = dt.hour
                        
                        hour_reservations = df[
                            (df['date'].dt.date == date) & 
                            (df['heure_debut_decimal'] <= hour) & 
                            (df['heure_fin_decimal'] > hour)
                        ]
                        
                        # Count this division's reservations
                        division_count = sum(hour_reservations['division'] == division) if not hour_reservations.empty else 0
                        
                        # Add to the dataframe
                        result_df.loc[dt, f'{division_name}_count'] = division_count
            
            print(f"DataFrame room occupation time created: {len(result_df)} hours")
            return result_df
        else:
            print("Warning: Required columns 'date', 'start_time', or 'end_time' not found in allocation data.")
            print(f"Available columns: {df.columns.tolist()}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error while processing room allocation data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    

def plot_correlation_matrix(data, title="Correlation Matrix"):
    """
    Create and display a correlation matrix for the input data.
    
    Parameters:
        data (DataFrame): Input dataframe containing variables to correlate
        title (str): Plot title
    """
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_data.corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 10))
    
    # Create a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap=cmap, 
        vmax=1, 
        vmin=-1, 
        center=0,
        square=True, 
        linewidths=.5, 
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return corr_matrix


def plot_feature_pairplot(data, target_col='pv_output', sample_size=None):
    """
    Create a pairplot to visualize relationships between features and the target variable.
    
    Parameters:
        data (DataFrame): Input dataframe containing features and target
        target_col (str): Name of the target column
        sample_size (int, optional): Number of samples to use (for large datasets)
    """
    # Create a copy of the data
    plot_data = data.copy()
    
    # If the dataset is large, sample it to speed up plotting
    if sample_size is not None and sample_size < len(plot_data):
        plot_data = plot_data.sample(sample_size, random_state=42)
        print(f"Using {sample_size} samples for pairplot visualization.")
    
    # Select only the most important numeric columns to avoid cluttered plots
    # Get columns that are likely to be important for prediction
    key_cols = ['pred_glob_ctrl', 'pred_t_2m_ctrl', 'temperature', 'hour_sin', 'day_of_year']
    key_cols = [col for col in key_cols if col in plot_data.columns]
    
    # Always include the target column
    if target_col not in key_cols and target_col in plot_data.columns:
        key_cols.append(target_col)
    
    # Check if we have enough columns
    if len(key_cols) < 2:
        print("Not enough columns for pairplot. Using all numeric columns.")
        key_cols = plot_data.select_dtypes(include=[np.number]).columns.tolist()
        if len(key_cols) > 6:  # Limit to 6 columns to avoid excessive plotting time
            key_cols = key_cols[:6]
    
    # Create the pairplot
    sns.set(style="ticks")
    g = sns.pairplot(
        plot_data[key_cols], 
        hue=target_col if target_col in plot_data.columns else None,
        corner=True,  # Only show the lower triangle
        diag_kind="kde",
        markers=".",
        height=2.5,
        plot_kws={'alpha': 0.6, 's': 30}
    )
    
    g.fig.suptitle('Feature Relationships Pairplot', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return g

def load_academic_calendar():
    """
    Creates a dataframe with official academic calendar information for 2022-2023.
    """
    try:
        calendar_periods = [
            {'start': '2022-09-19', 'end': '2022-12-20', 'type': 'autumn_semester'},
            {'start': '2022-12-21', 'end': '2023-01-04', 'type': 'christmas_holiday'},
            {'start': '2023-01-05', 'end': '2023-01-17', 'type': 'autumn_semester'},
            {'start': '2023-01-18', 'end': '2023-01-28', 'type': 'exam_period'},
            {'start': '2023-01-29', 'end': '2023-02-19', 'type': 'between_semesters'},
            {'start': '2023-02-20', 'end': '2023-04-06', 'type': 'spring_semester'},
            {'start': '2023-04-07', 'end': '2023-04-16', 'type': 'easter_holiday'},
            {'start': '2023-04-17', 'end': '2023-05-17', 'type': 'spring_semester'},
            {'start': '2023-05-18', 'end': '2023-05-20', 'type': 'holiday'},
            {'start': '2023-05-21', 'end': '2023-06-20', 'type': 'spring_semester'},
            {'start': '2023-06-21', 'end': '2023-07-01', 'type': 'exam_period'},
            {'start': '2023-07-02', 'end': '2023-09-18', 'type': 'summer_break'}
        ]
        
        start_date = '2022-09-01'
        end_date = '2023-09-30'
        
        all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        calendar_df = pd.DataFrame(index=all_dates)
        calendar_df.index.name = 'date'
        
        calendar_df['is_semester_period'] = 0
        calendar_df['is_autumn_semester'] = 0
        calendar_df['is_spring_semester'] = 0
        calendar_df['is_holiday_period'] = 0
        calendar_df['is_christmas_holiday'] = 0
        calendar_df['is_easter_holiday'] = 0
        calendar_df['is_summer_break'] = 0
        calendar_df['is_between_semesters'] = 0
        calendar_df['is_exam_period'] = 0
        calendar_df['is_ascension'] = 0
        
        for period in calendar_periods:
            start = pd.Timestamp(period['start'])
            end = pd.Timestamp(period['end'])
            period_type = period['type']
            
            mask = (calendar_df.index >= start) & (calendar_df.index <= end)
            
            if period_type == 'autumn_semester':
                calendar_df.loc[mask, 'is_semester_period'] = 1
                calendar_df.loc[mask, 'is_autumn_semester'] = 1
            elif period_type == 'spring_semester':
                calendar_df.loc[mask, 'is_semester_period'] = 1
                calendar_df.loc[mask, 'is_spring_semester'] = 1
            elif period_type == 'christmas_holiday':
                calendar_df.loc[mask, 'is_holiday_period'] = 1
                calendar_df.loc[mask, 'is_christmas_holiday'] = 1
            elif period_type == 'easter_holiday':
                calendar_df.loc[mask, 'is_holiday_period'] = 1
                calendar_df.loc[mask, 'is_easter_holiday'] = 1
            elif period_type == 'summer_break':
                calendar_df.loc[mask, 'is_holiday_period'] = 1
                calendar_df.loc[mask, 'is_summer_break'] = 1
            elif period_type == 'between_semesters':
                calendar_df.loc[mask, 'is_between_semesters'] = 1
            elif period_type == 'exam_period':
                calendar_df.loc[mask, 'is_exam_period'] = 1
            elif period_type == 'holiday':
                calendar_df.loc[mask, 'is_holiday_period'] = 1
                if start == pd.Timestamp('2023-05-18'):
                    calendar_df.loc[mask, 'is_ascension'] = 1
        
        holidays = {
            '2022-11-01': 'Toussaint',
            '2022-12-08': 'Immaculée conception',
            '2023-05-29': 'Lundi de Pentecôte',
            '2023-06-08': 'Fête Dieu'
        }
        
        calendar_df['is_holiday'] = 0
        calendar_df['holiday_name'] = ''
        
        for date, name in holidays.items():
            calendar_df.loc[date, 'is_holiday'] = 1
            calendar_df.loc[date, 'holiday_name'] = name
            calendar_df.loc[date, 'is_holiday_period'] = 1
        
        calendar_df['is_weekend'] = (calendar_df.index.dayofweek >= 5).astype(int)
        
       
        autumn_weeks = {
            'A0': ('2022-09-12', '2022-09-17'),
            'A1': ('2022-09-19', '2022-09-24'),
            'A2': ('2022-09-26', '2022-10-01'),
            'A3': ('2022-10-03', '2022-10-08'),
            'A4': ('2022-10-10', '2022-10-15'),
            'A5': ('2022-10-17', '2022-10-22'),
            'A6': ('2022-10-24', '2022-10-29'),
            'A7': ('2022-10-31', '2022-11-05'),
            'A8': ('2022-11-07', '2022-11-12'),
            'A9': ('2022-11-14', '2022-11-19'),
            'A10': ('2022-11-21', '2022-11-26'),
            'A11': ('2022-11-28', '2022-12-03'),
            'A12': ('2022-12-05', '2022-12-10'),
            'A13': ('2022-12-12', '2022-12-17'),
            'A14': ('2022-12-19', '2022-12-24'),
            'A15': ('2023-01-02', '2023-01-07'),
            'A16': ('2023-01-09', '2023-01-14')
        }
        
        spring_weeks = {
            'P1': ('2023-02-20', '2023-02-25'),
            'P2': ('2023-02-27', '2023-03-04'),
            'P3': ('2023-03-06', '2023-03-11'),
            'P4': ('2023-03-13', '2023-03-18'),
            'P5': ('2023-03-20', '2023-03-25'),
            'P6': ('2023-03-27', '2023-04-01'),
            'P7': ('2023-04-03', '2023-04-08'),
            'P8': ('2023-04-10', '2023-04-15'),
            'P9': ('2023-04-24', '2023-04-29'),
            'P10': ('2023-05-01', '2023-05-06'),
            'P11': ('2023-05-08', '2023-05-13'),
            'P12': ('2023-05-15', '2023-05-20'),
            'P13': ('2023-05-22', '2023-05-27'),
            'P14': ('2023-05-29', '2023-06-03'),
            'P15': ('2023-06-05', '2023-06-10'),
            'P16': ('2023-06-12', '2023-06-17')
        }
        
        calendar_df['academic_week'] = ''
        
        for week_code, (start, end) in autumn_weeks.items():
            mask = (calendar_df.index >= pd.Timestamp(start)) & (calendar_df.index <= pd.Timestamp(end))
            calendar_df.loc[mask, 'academic_week'] = week_code
        
        for week_code, (start, end) in spring_weeks.items():
            mask = (calendar_df.index >= pd.Timestamp(start)) & (calendar_df.index <= pd.Timestamp(end))
            calendar_df.loc[mask, 'academic_week'] = week_code
        
        return calendar_df
        
    except Exception as e:
        print(f"Erreur lors de la création du calendrier académique: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# ----------------------------------
# Functions to prepare data
# ----------------------------------

def add_enhanced_time_features(df):
    
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Error when converting index : {e}")
            return df
    
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['day_of_week'] = df.index.dayofweek
    df['week_of_year'] = df.index.isocalendar().week
    df['day_of_year'] = df.index.dayofyear
    
    df['is_day_hours'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
    df['is_evening_hours'] = ((df['hour'] >= 17) & (df['hour'] < 22)).astype(int)
    df['is_night_hours'] = ((df['hour'] < 8) | (df['hour'] >= 22)).astype(int)
    
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_working_day'] = (~df.index.dayofweek.isin([5, 6])).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    
    df['is_morning_class'] = ((df['hour'] >= 8) & (df['hour'] < 12) & (df['is_working_day'] == 1)).astype(int)
    df['is_afternoon_class'] = ((df['hour'] >= 13) & (df['hour'] < 17) & (df['is_working_day'] == 1)).astype(int)
    df['is_evening_class'] = ((df['hour'] >= 17) & (df['hour'] < 21) & (df['is_working_day'] == 1)).astype(int)
    
    
    season_dict = {
        1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 
        5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer', 
        9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter'
    }
    df['season'] = df['month'].map(season_dict)
    
    
    df['is_winter'] = (df['season'] == 'winter').astype(int)
    df['is_spring'] = (df['season'] == 'spring').astype(int)
    df['is_summer'] = (df['season'] == 'summer').astype(int)
    df['is_autumn'] = (df['season'] == 'autumn').astype(int)
    
    df['is_building_open'] = ((df['hour'] >= 7) & (df['hour'] < 22) & (df['is_working_day'] == 1)).astype(int)
    df['is_building_closed'] = 1 - df['is_building_open']
    
    return df

def merge_hourly_datasets(consumption_df, weather_df, room_df, calendar_df):
    """
    Merges dataframes with hourly granularity
    """
    print("Merging data with hour granularity...")
    
    if consumption_df.empty:
        print("Error : consumption data missing")
        return pd.DataFrame()
    
    result = consumption_df.copy()
    
    if not weather_df.empty:
        print(f"merge with meteo data ({weather_df.shape})...")
        
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.to_datetime(result.index)
            except:
                print("Error: Impossible to convert index consumption into DatetimeIndex.")
                return pd.DataFrame()
        
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            try:
                weather_df.index = pd.to_datetime(weather_df.index)
            except:
                print("Error : Impossible to convert index meteo into DatetimeIndex.")
                return pd.DataFrame()
        
       
        
        # Fusion
        result = pd.merge(
            result,
            weather_df,
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_weather')
        )
        
        if not isinstance(result.index, pd.DatetimeIndex):
            try:
                result.index = pd.to_datetime(result.index)
            except:
                print("Error: Impossible to convert index into DatetimeIndex after merging meteo")
                return pd.DataFrame()
        
        for col in weather_df.columns:
            if col in result.columns and result[col].isna().any():
                missing_before = result[col].isna().sum()
                
                try:
                    result[col] = result[col].interpolate(method='linear').bfill().ffill()
                except Exception as e:
                    result[col] = result[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
                
                missing_after = result[col].isna().sum()
                print(f"  {col}: {missing_before} -> {missing_after} missing value after interpolation")
    
    if not room_df.empty:
        print(f"Fusion with room occupency ({room_df.shape})...")

        
        if not isinstance(room_df.index, pd.DatetimeIndex):
            try:
                room_df.index = pd.to_datetime(room_df.index)
                print(f"Index room occupency converted to DatetimeIndex")
            except:
                print("Error: Impossible to convert room ocupency index to DatetimeIndex.")
        
        if isinstance(room_df.index, pd.DatetimeIndex):
            
            
            result = pd.merge(
                result,
                room_df,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_room')
            )
            
            if not isinstance(result.index, pd.DatetimeIndex):
                try:
                    result.index = pd.to_datetime(result.index)
                except:
                    print("Error: Impossible to convert index result into DatetimeIndex.")
            
            room_cols = room_df.columns
            for col in room_cols:
                if col in result.columns and result[col].isna().any():
                    missing_before = result[col].isna().sum()
                    
                    if 'occupied' in col or 'occupation' in col or 'rooms' in col:
                        result[col] = result[col].fillna(0)
                    else:
                        result[col] = result[col].interpolate(method='linear').bfill().ffill().fillna(0)
                    
                    missing_after = result[col].isna().sum()
                    print(f"  {col}: {missing_before} -> {missing_after} missing value after treatment")
    
    result = add_enhanced_time_features(result)
    
  
    if not calendar_df.empty:
        print(f"Add calendar info...")
        
       
        result['datetime'] = result.index
       
        result['date'] = result['datetime'].dt.strftime('%Y-%m-%d')
        
       
        calendar_df = calendar_df.reset_index()
        calendar_df['date'] = calendar_df['date'].dt.strftime('%Y-%m-%d')
        
        print(f"Type of column 'date' in result: {result['date'].dtype}")
        print(f"Type of column 'date' in calendar_df: {calendar_df['date'].dtype}")
        
       
        result = pd.merge(
            result,
            calendar_df,
            on='date',
            how='left',
            suffixes=('', '_cal')
        )
        
        result = result.set_index('datetime')
        result = result.drop(columns=['date'])



    result.index = pd.to_datetime(result.index)  
    print("in merge_hourly_datasets, final index type :", type(result.index))
    print("Final index :", result.index[:5])  

    
    if not isinstance(result.index, pd.DatetimeIndex):
        try:
            result.index = pd.to_datetime(result.index)
            print("Index converted to DatetimeIndex after all merges")
        except:
            print("Warning : Index not converted to DatetimeIndex after all merges")
        
    cal_cols = [col for col in calendar_df.columns if col != 'date'] if not calendar_df.empty else []
    for col in cal_cols:
        if col in result.columns and result[col].isna().any():
            if col.startswith('is_'):
                result[col] = result[col].fillna(0)
            elif result[col].dtype == 'object':
                result[col] = result[col].fillna('')
            else:
                result[col] = result[col].interpolate(method='pad')
    
    if 'temperature' in result.columns and 'is_building_open' in result.columns:
        result['is_cold_and_occupied'] = ((result['temperature'] < 10) & (result['is_building_open'] == 1)).astype(int)
    
    if 'is_weekend' in result.columns and 'is_holiday_period' in result.columns:
        result['is_weekend_or_holiday'] = ((result['is_weekend'] == 1) | (result['is_holiday_period'] == 1)).astype(int)
    
    if 'rooms_occupied' in result.columns:
        try:
            if not result['rooms_occupied'].isna().all():
                result['is_high_occupation'] = (result['rooms_occupied'] > result['rooms_occupied'].quantile(0.75)).astype(int)
                result['is_low_occupation'] = (result['rooms_occupied'] < result['rooms_occupied'].quantile(0.25)).astype(int)
            else:
                result['is_high_occupation'] = 0
                result['is_low_occupation'] = 0
        except Exception as e:
            print(f"Error when creating room occupency features: {e}")
            result['is_high_occupation'] = 0
            result['is_low_occupation'] = 0
        
        if 'temperature' in result.columns:
            try:
                result['temp_occupation_interaction'] = result['temperature'] * result['rooms_occupied']
            except Exception as e:
                print(f"Erreur lors de la création de l'interaction temp-occupation: {e}")
                
                result['temp_occupation_interaction'] = 0
    
    missing_values = result.isna().sum()
    cols_with_missing = missing_values[missing_values > 0]
    
    if not cols_with_missing.empty:
        print("Missing values after all merges:")
        for col, count in cols_with_missing.items():
            print(f"  {col}: {count} missing values")
            
            try:
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].interpolate(method='linear').bfill().ffill().fillna(0)
                else:
                    if result[col].dtype == 'object' or pd.api.types.is_categorical_dtype(result[col]):
                        mode_value = result[col].mode()[0] if not result[col].mode().empty else ''
                        result[col] = result[col].fillna(mode_value)
            except Exception as e:
                print(f"Erreur lors du traitement des valeurs manquantes pour {col}: {e}")
                if pd.api.types.is_numeric_dtype(result[col]):
                    result[col] = result[col].fillna(0)
                else:
                    result[col] = result[col].fillna('')
    
    print(f"final after merge and processing: {result.shape}")
    return result

# ----------------------------------
# Functions for modeling
# ----------------------------------

def train_test_split_timeseries_robust(df, test_size=0.2):
    """
    Splits data into training and test sets chronologically,
    with robust handling of small datasets.
    """
    # Check if DataFrame is empty
    if df.empty:
        print("Warning: Empty DataFrame for chronological split")
        empty_df = pd.DataFrame(columns=df.columns)
        return empty_df, empty_df
    
    # Sort by time index
    df_sorted = df.sort_index()
    n = len(df_sorted)
    
    # Special case: very small datasets
    if n <= 2:
        print(f"WARNING: Only {n} samples available!")
        # If only one sample, use it for both training and testing
        if n == 1:
            return df_sorted.copy(), df_sorted.copy()
        # If two samples, use first for training, second for testing
        else:
            return df_sorted.iloc[:1].copy(), df_sorted.iloc[1:].copy()
    
    # Calculate split indices with minimum protection
    test_samples = max(1, int(n * test_size))
    train_samples = n - test_samples
    
    # Ensure there's at least one sample in each set
    if train_samples == 0:
        train_samples = 1
        test_samples = n - 1
    
    # Split the data
    train_df = df_sorted.iloc[:train_samples].copy()
    test_df = df_sorted.iloc[train_samples:].copy()
    
    print(f"Chronological split: train={len(train_df)}, test={len(test_df)}")
    
    return train_df, test_df

def create_advanced_model(X_train, y_train, model_type='xgboost'):
    """
    Creates an advanced model for energy consumption prediction.
    Handles categorical columns correctly.
    """
    print(f"Creating an advanced model of type '{model_type}'...")
    
    if X_train.empty or len(y_train) == 0:
        print("ERROR: Empty training data.")
        return None
    
    # Data preprocessing: converting object type columns
    X_train_processed = X_train.copy()
    
    # Identify columns of type 'object'
    object_columns = X_train_processed.select_dtypes(include=['object']).columns.tolist()
    if object_columns:
        print(f"Converting 'object' type columns: {object_columns}")
        X_train_processed = preprocess_object_columns(X_train_processed, object_columns)
    
    # Model options
    if model_type.lower() == 'xgboost':
        # XGBoost - Excellent for tabular data, handles missing values well
        print("Configuring XGBoost model...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=1,
            gamma=0,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
    elif model_type.lower() == 'lightgbm':
        # LightGBM - Faster than XGBoost, good general performance
        print("Configuring LightGBM model...")
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=7,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='regression',
            random_state=42,
            n_jobs=-1
        )
        
    elif model_type.lower() == 'randomforest':
        # RandomForest - Robust and less sensitive to hyperparameter tuning
        print("Configuring RandomForest model...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
    else:
        print(f"Model type {model_type} not recognized. Using XGBoost by default.")
        model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=7, 
            random_state=42
        )
    
    # Define a pipeline with scaling for certain models
    if model_type.lower() in ['xgboost', 'lightgbm']:
        # These models don't need scaling
        pipeline = Pipeline([
            ('model', model)
        ])
    else:
        # For RandomForest and others, apply scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
    
    # Train the model
    print(f"Training the model on {len(X_train_processed)} examples with {X_train_processed.shape[1]} features...")
    pipeline.fit(X_train_processed, y_train)
    
    # Extract the model from the pipeline
    if model_type.lower() in ['xgboost', 'lightgbm', 'randomforest']:
        # These models have a feature_importances_ property
        trained_model = pipeline.named_steps['model']
        
        # Calculate feature importance
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,  # Use original column names
            'Importance': trained_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
    else:
        # For other models, we might not have feature importance
        feature_importance = pd.DataFrame()
    
    # Also store the data preprocessing function for future predictions
    preprocess_func = lambda X: X.copy().astype({col: 'category' for col in object_columns})
    
    return {
        'model': pipeline,
        'feature_importance': feature_importance,
        'model_type': model_type,
        'object_columns': object_columns,
    }

def evaluate_advanced_model(model_info, X_test, y_test):
    """
    Evaluates an advanced model on test data.
    Provides detailed metrics and visualizations.
    """
    if model_info is None or X_test.empty or len(y_test) == 0:
        print("WARNING: Cannot evaluate the model (missing data)")
        return None
    
    pipeline = model_info['model']
    
    # Preprocess test data the same way as training data
    X_test_processed = X_test.copy()
    
    # Apply the same transformations to object type columns
    object_columns = model_info.get('object_columns', [])
    for col in object_columns:
        if col in X_test_processed.columns:
            # If the column is empty or contains only one unique value, replace with 0
            if X_test_processed[col].nunique() <= 1:
                X_test_processed[col] = 0
            # Otherwise, convert to category then to numeric codes
            else:
                X_test_processed[col] = X_test_processed[col].astype('category').cat.codes
    
    # Generate predictions
    y_pred = pipeline.predict(X_test_processed)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Avoid division by zero in MAPE
    mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), 0.01))) * 100
    
    print(f"\nEvaluation metrics on {len(y_test)} test examples:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # Prepare evaluation results
    eval_results = {
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        },
        'predictions': y_pred,
        'actual': y_test
    }
    
    return eval_results

def visualize_predictions(eval_results, title="Improved Consumption Prediction"):
    """
    Visualizes prediction results in detail.
    """
    if eval_results is None or 'predictions' not in eval_results:
        print("ERROR: Evaluation results not available for visualization.")
        return
    
    y_test = eval_results['actual']
    y_pred = eval_results['predictions']
    
    # For time series data
    if isinstance(y_test.index, pd.DatetimeIndex):
        # 1. Create a DataFrame for easier visualization
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Prediction': y_pred
        }, index=y_test.index)
        
        # 2. Global view of predictions
        plt.figure(figsize=(15, 8))
        plt.plot(results_df.index, results_df['Actual'], label='Actual Consumption', color='blue', alpha=0.7)
        plt.plot(results_df.index, results_df['Prediction'], label='Prediction', color='red', alpha=0.7)
        plt.title(f"{title} - Global View", fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Consumption', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 3. Detailed view of one week (if available)
        if len(results_df) > 24*7:
            # Select the last week
            last_week = results_df.iloc[-24*7:]
            
            plt.figure(figsize=(15, 8))
            plt.plot(last_week.index, last_week['Actual'], label='Actual Consumption', color='blue', alpha=0.7)
            plt.plot(last_week.index, last_week['Prediction'], label='Prediction', color='red', alpha=0.7)
            plt.title(f"{title} - Last Week", fontsize=15)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Consumption', fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # 4. Daily aggregation
        daily_results = results_df.groupby(results_df.index.date).sum()
        
        plt.figure(figsize=(15, 8))
        plt.plot(daily_results.index, daily_results['Actual'], label='Actual Consumption', marker='o', color='blue')
        plt.plot(daily_results.index, daily_results['Prediction'], label='Prediction', marker='x', color='red')
        plt.title(f"{title} - Daily Aggregation", fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Daily Consumption', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # 5. Error distribution
        errors = y_test - y_pred
        
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(errors, kde=True)
        plt.title('Error Distribution', fontsize=15)
        plt.xlabel('Prediction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=errors)
        plt.title('Error Boxplot', fontsize=15)
        plt.xlabel('Prediction Error', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        # 6. Scatter plot (actual vs predicted values)
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f"{title} - Actual vs Predicted Values", fontsize=15)
        plt.xlabel('Actual Consumption', fontsize=12)
        plt.ylabel('Predicted Consumption', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    else:
        # For non-temporal data
        # Simplified scatter plot
        plt.figure(figsize=(10, 10))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f"{title} - Actual vs Predicted Values", fontsize=15)
        plt.xlabel('Actual Consumption', fontsize=12)
        plt.ylabel('Predicted Consumption', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def visualize_feature_importance(model_info, top_n=20):
    """
    Visualizes feature importance for the model.
    """
    if model_info is None or 'feature_importance' not in model_info or model_info['feature_importance'].empty:
        print("ERROR: Feature importance information not available.")
        return
    
    # Get the N most important features
    importance_df = model_info['feature_importance'].head(top_n)
    
    # Create a more detailed visualization
    plt.figure(figsize=(12, 10))
    
    # Create a horizontal bar chart
    bars = plt.barh(
        importance_df['Feature'][::-1],
        importance_df['Importance'][::-1],
        color=plt.cm.viridis(np.linspace(0, 0.8, len(importance_df)))
    )
    
    # Add values at the end of the bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + bar.get_width() * 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{importance_df["Importance"].iloc[-(i+1)]:.4f}',
            va='center'
        )
    
    # Title and labels
    model_type = model_info.get('model_type', 'Advanced')
    plt.title(f'Top {top_n} Most Important Features ({model_type} Model)', fontsize=15)
    plt.xlabel('Relative Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    
    # Improve appearance
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


    

# ----------------------------------
# Main hourly model function
# ----------------------------------

def run_improved_hourly_model(data_date=None, model_type='xgboost', save_outputs=True, visualization=False):
    print(f"Starting improved hourly prediction model with {model_type}")
    print("=" * 60)

    # 1. Get directories
    dirs = get_directories(data_date)

    # 2. Calcul du cutoff_dt (il faut le faire avant d’en avoir besoin)
    if data_date is None:
        data_date = datetime.today().strftime('%Y-%m-%d')
    cutoff_dt = datetime.strptime(data_date, '%Y-%m-%d')

    # 3. Fallback si pas de dossier de conso trouvé
    if not dirs['all_conso_dirs']:
        print("WARNING: No consumption dirs found before cutoff date. Using most recent available directory as fallback.")
        base = dirs['base_dir']
        all_cleaned = sorted(
            glob.glob(os.path.join(base, 'cleaned_data_*')),
            key=lambda p: datetime.strptime(
                re.search(r'cleaned_data_(\d{4}-\d{2}-\d{2})$', p).group(1),
                '%Y-%m-%d'
            )
        )
        if all_cleaned:
            fallback = os.path.join(all_cleaned[0], 'BellevueConso')
            print(f"Including fallback dir: {fallback}")
            dirs['all_conso_dirs'] = [fallback]
        else:
            print("ERROR: No cleaned_data_* directories found at all.")
            return None

    # 4. Chargement de la conso
    consumption_df = load_consumption_data(
        dirs['all_conso_dirs'],
        cutoff_date=cutoff_dt
    )

    # 3) Chargement des prédictions météo depuis *tous* les dossiers filtrés
    weather_df = load_prediction_weather_data(
        dirs['all_meteo_dirs'],
        cutoff_date=cutoff_dt
    )

    # 4) Pour les salles, on prend le dernier dossier (le plus récent avant cutoff)
    if dirs['all_room_dirs']:
        latest_room_dir = dirs['all_room_dirs'][-1]
    else:
        latest_room_dir = dirs['room_dir']
    room_df = load_room_allocation_data_hourly(latest_room_dir)
    
    # Load academic calendar
    calendar_df = load_academic_calendar()
    
    # 2. Merge data with the improved function for hourly granularity
    merged_data = merge_hourly_datasets(consumption_df, weather_df, room_df, calendar_df)
    
    if merged_data.empty:
        print("ERROR: Failed to merge data. Empty dataset.")
        return None
    
    if visualization:
    
        print("\nGenerating correlation matrix...")
        corr_matrix = plot_correlation_matrix(merged_data, title="Energy Consumption - Features Correlation Matrix")
        
        if 'consumption' in merged_data.columns:
            consumption_corrs = corr_matrix['consumption'].sort_values(ascending=False)
            print("\nTop features correlated with consumption:")
            print(consumption_corrs.head(10))
            print("\nBottom features correlated with consumption:")
            print(consumption_corrs.tail(5))
        
        print("\nGenerating pairplot of key features...")
        plot_feature_pairplot(merged_data, target_col='consumption', sample_size=1000)



    
    # 3. Prepare features and target at hourly granularity
    X = merged_data.copy()
    y = X['consumption'].copy()
    X = X.drop(columns=['consumption'])
    
    print(f"Hourly data: {len(X)} records, {X.shape[1]} features")
    
    # 4. Chronological split and training on hourly data
    X_train, X_test = train_test_split_timeseries_robust(X)
    y_train, y_test = train_test_split_timeseries_robust(y)
    
    # Inform about the duration of training and test sets
    if isinstance(X_train.index, pd.DatetimeIndex) and len(X_train) > 0:
        train_days = (X_train.index.max() - X_train.index.min()).days + 1
        print(f"Training set: {len(X_train)} hours over approximately {train_days} days")
    
    if isinstance(X_test.index, pd.DatetimeIndex) and len(X_test) > 0:
        test_days = (X_test.index.max() - X_test.index.min()).days + 1
        print(f"Test set: {len(X_test)} hours over approximately {test_days} days")
    
    # 5. Train the model at hourly granularity
    model_info = create_advanced_model(X_train, y_train, model_type=model_type)
    
    if model_info is None:
        print("ERROR: Model training failed.")
        return None
    
    # 6. Evaluation at hourly granularity
    hourly_eval_results = evaluate_advanced_model(model_info, X_test, y_test)
    
    if hourly_eval_results is None:
        print("ERROR: Hourly model evaluation failed.")
        return None
    
    # 7. Aggregate hourly predictions into daily predictions for comparison
    print("\nAggregating hourly predictions into daily predictions...")

    # Ensure the index is a DatetimeIndex
    if not isinstance(hourly_eval_results['actual'].index, pd.DatetimeIndex):
        print("Converting index to DatetimeIndex for aggregation...")
        try:
            actual_index = pd.to_datetime(hourly_eval_results['actual'].index)
            predicted_index = pd.to_datetime(hourly_eval_results['actual'].index)  # Same index to preserve alignment
        except:
            print("Creating an artificial time index...")
            base_date = pd.Timestamp('2023-01-01')
            dates = [base_date + pd.Timedelta(hours=i) for i in range(len(hourly_eval_results['actual']))]
            actual_index = pd.DatetimeIndex(dates)
            predicted_index = actual_index
    else:
        actual_index = hourly_eval_results['actual'].index
        predicted_index = actual_index

    # Create a DataFrame with correctly indexed series
    hourly_results = pd.DataFrame({
        'actual': hourly_eval_results['actual'].values,
        'predicted': hourly_eval_results['predictions']
    }, index=actual_index)

    # Use resample for daily aggregation (robust method)
    daily_results = hourly_results.resample('D').sum()

    # Calculate daily metrics
    daily_actual = daily_results['actual']
    daily_pred = daily_results['predicted']

    daily_rmse = np.sqrt(mean_squared_error(daily_actual, daily_pred))
    daily_mae = mean_absolute_error(daily_actual, daily_pred)
    daily_r2 = r2_score(daily_actual, daily_pred)
    daily_mape = np.mean(np.abs((daily_actual - daily_pred) / np.maximum(np.abs(daily_actual), 0.01))) * 100
    
    print(f"\nMetrics after daily aggregation:")
    print(f"  Daily RMSE: {daily_rmse:.4f}")
    print(f"  Daily MAE: {daily_mae:.4f}")
    print(f"  Daily R²: {daily_r2:.4f}")
    print(f"  Daily MAPE: {daily_mape:.2f}%")
    
    # Add aggregated results to evaluation results
    hourly_eval_results['daily_metrics'] = {
        'rmse': daily_rmse,
        'mae': daily_mae,
        'r2': daily_r2,
        'mape': daily_mape
    }
    hourly_eval_results['daily_data'] = {
        'actual': daily_actual,
        'predicted': daily_pred
    }

    if visualization :
    
        # 8. Visualizations (including hourly vs daily comparisons)
        print("\n8. Creating visualizations...")
        
        # Visualize hourly predictions
        visualize_predictions(hourly_eval_results, title=f"Hourly Consumption Prediction ({model_type})")
        
        # Visualize feature importance
        visualize_feature_importance(model_info, top_n=15)
        
        # 9. Compare hourly vs daily metrics
        print("\n9. Comparison of hourly vs daily performance:")
        
        metrics_comparison = pd.DataFrame({
            'Hourly': [
                hourly_eval_results['metrics']['rmse'],
                hourly_eval_results['metrics']['mae'],
                hourly_eval_results['metrics']['r2'],
                hourly_eval_results['metrics']['mape']
            ],
            'Daily (aggregated)': [
                hourly_eval_results['daily_metrics']['rmse'],
                hourly_eval_results['daily_metrics']['mae'],
                hourly_eval_results['daily_metrics']['r2'],
                hourly_eval_results['daily_metrics']['mape']
            ]
        }, index=['RMSE', 'MAE', 'R²', 'MAPE (%)'])
        
        print(metrics_comparison)
    
    # 10. Create prediction function that can generate hourly predictions and aggregate them
    print("\n10. Creating hourly prediction function with flexible aggregation...")
    
    
    # 11. Save model and results if requested
    if save_outputs:
        print("\n11. Saving model and predictions...")
        
        # Create output directories
        output_dir = 'predictions'
        model_dir = 'models'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # CORRECTION: Save hourly predictions with preserved index
        hourly_results = pd.DataFrame({
            'actual': hourly_eval_results['actual'],
            'predicted': hourly_eval_results['predictions']
        }, index=hourly_eval_results['actual'].index)
        hourly_csv_path = os.path.join(output_dir, f"hourly_{model_type}_predictions_{current_date}.csv")
        hourly_results.to_csv(hourly_csv_path, date_format='%Y-%m-%d %H:%M:%S')
        
        # CORRECTION: Save daily predictions with preserved index
        daily_results = pd.DataFrame({
            'actual': hourly_eval_results['daily_data']['actual'],
            'predicted': hourly_eval_results['daily_data']['predicted']
        }, index=hourly_eval_results['daily_data']['actual'].index)
        daily_csv_path = os.path.join(output_dir, f"daily_aggregated_{model_type}_predictions_{current_date}.csv")
        daily_results.to_csv(daily_csv_path, date_format='%Y-%m-%d')

        if 'preprocess_func' in model_info:
            del model_info['preprocess_func']
        
        # Save the model
        model_path = os.path.join(model_dir, "consumption_model.joblib")
        dump(model_info, model_path)
        
        # Save model metadata 
        metadata = {
            'model_type': model_info.get('model_type', model_type),
            'granularity': 'hourly',
            'features': list(X_train.columns),
            'hourly_metrics': hourly_eval_results['metrics'],
            'daily_metrics': hourly_eval_results['daily_metrics'],
            'timestamp': current_date,
            'data_shape': {
                'X_train': X_train.shape,
                'X_test': X_test.shape
            }
        }
        
        metadata_path = os.path.join(model_dir, f"hourly_{model_type}_metadata_{current_date}.json")
        with open(metadata_path, 'w') as f:
            # Convert numpy values to standard Python types
            def convert_numpy(obj):
                if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                elif isinstance(obj, (tuple, list)):
                    return [convert_numpy(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                return obj
            
            json.dump(convert_numpy(metadata), f, indent=2)
        
        print(f"Hourly predictions saved to: {hourly_csv_path}")
        print(f"Aggregated daily predictions saved to: {daily_csv_path}")
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
    
    # 12. Return results and prediction function
    return {
    'model': model_info,
    'metrics': {
        'hourly': hourly_eval_results['metrics'],
        'daily': hourly_eval_results['daily_metrics']
    },
    'evaluation': {
        'hourly': hourly_eval_results,
        'data': {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    }
}


def preprocess_object_columns(X, object_columns):
    """Preprocessing for 'object' type columns"""
    X_processed = X.copy()
    for col in object_columns:
        if col in X_processed.columns:
            # If the column is empty or contains only one unique value, replace with 0
            if X_processed[col].nunique() <= 1:
                X_processed[col] = 0
            # Otherwise, convert to category then to numeric codes
            else:
                X_processed[col] = X_processed[col].astype('category').cat.codes
    return X_processed

# Main function to run the model
def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved hourly energy consumption prediction model')
    parser.add_argument('--date', type=str, default=None, 
                      help='Data date in YYYY-MM-DD format (default: current date)')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'lightgbm', 'randomforest'],
                      help='Type of model to use (default: xgboost)')
    parser.add_argument('--save', action='store_true', default=True,
                      help='Save results (default: True)')
    parser.add_argument('--no-save', dest='save', action='store_false',
                      help='Do not save results')
    
    
    args = parser.parse_args()
    
    print(f"Configuration:")
    print(f"  Date: {args.date if args.date else 'Current'}")
    print(f"  Model: {args.model}")
    print(f"  Save: {args.save}")
    print(f"  Future days: {args.future_days}")
    print(f"  Generate future predictions: {args.generate_future}")
    
    results = run_improved_hourly_model(
        data_date=args.date,
        model_type=args.model,
        save_outputs=args.save,
        visualization=False
    )
    
# Execute if called directly
if __name__ == "__main__":
    try:
        print("Starting improved hourly energy consumption prediction model...")
        results = main()
        print("\nExecution completed successfully.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()