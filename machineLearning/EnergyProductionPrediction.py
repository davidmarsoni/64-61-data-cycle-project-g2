# Solar Energy Production Prediction Model
# ----------------------------------------
# This model predicts solar panel energy production based on weather forecasts.
# It uses the predicted solar radiation (PRED_GLOB_ctrl), temperature (PRED_T_2M_ctrl),
# and other relevant features to predict PV output.

import pandas as pd
import numpy as np
import os
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.model_selection import cross_val_score
import re
import pickle




# ---------------------------------
# Configuration
# ---------------------------------

def get_directories(data_date=None, use_all_folders=True):
    """
    Get directory paths for data files.
    
    Parameters:
        data_date (str): Date string in 'YYYY-MM-DD' format, defaults to today if None
        use_all_folders (bool): If True, gather paths from multiple date-specific folders
        
    Returns:
        dict: Dictionary of directory paths and lists of all relevant folders to scan
    """
    import os
    from datetime import datetime, timedelta
    import glob
    
    if data_date is None:
        data_date = datetime.today().strftime('%Y-%m-%d')
    
    base_dir = os.getenv('BASE_DIR', 'C:/DataCollection')
    
    # Get the specific data directory for today's date
    specific_data_dir = os.path.join(base_dir, f'cleaned_data_{data_date}')
    
    # Directories within the date-specific folder
    solar_dir = os.path.join(specific_data_dir, 'Solarlogs')
    meteo_dir = os.path.join(specific_data_dir, 'Meteo')
    
    if use_all_folders:
        # Find all available cleaned_data_* folders
        all_data_dirs = glob.glob(os.path.join(base_dir, 'cleaned_data_*'))
        all_data_dirs.sort()  # Sort them chronologically
        
        # Store paths to all available subdirectories
        all_solar_dirs = [os.path.join(data_dir, 'Solarlogs') for data_dir in all_data_dirs]
        all_meteo_dirs = [os.path.join(data_dir, 'Meteo') for data_dir in all_data_dirs]
      
        
        return {
            'base_dir': base_dir,
            'specific_data_dir': specific_data_dir,
            'solar_dir': solar_dir,
            'meteo_dir': meteo_dir,
            'all_data_dirs': all_data_dirs,
            'all_solar_dirs': all_solar_dirs,
            'all_meteo_dirs': all_meteo_dirs,
        }
    else:
        # Use only the specified date folder
        return {
            'base_dir': base_dir,
            'data_dir': specific_data_dir,
            'solar_dir': solar_dir,
            'meteo_dir': meteo_dir,
        }

# Modified load_filtered_csv_files function to recursively search subdirectories
def load_filtered_csv_files(directory, file_pattern=None, date_filter=None, cutoff_date=None, recursive=True):
    """
    Load multiple CSV files from a directory with optional pattern matching and date filtering.
    Filters files based on date in filename or folder name without reading file contents.
    
    Parameters:
        directory (str): Directory path containing CSV files
        file_pattern (str, optional): Pattern to match filenames (e.g., 'PV')
        date_filter (str, optional): Date to filter by (e.g., '2023-01-01')
        cutoff_date (datetime, optional): Exclude data after this date
        recursive (bool): Whether to search subdirectories recursively
        
    Returns:
        DataFrame: Combined data from all matching CSV files
    """

    
    all_data = []
    skipped_files = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist.")
        return pd.DataFrame()
    
    def extract_date_from_string(text):
        """
        Try to extract a date from a string (filename or folder name) using various patterns.
        Returns a datetime object if found, None otherwise.
        """
        # Common date patterns in filenames (add more as needed)
        patterns = [
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY
            r'(\d{8})',              # YYYYMMDD
            r'(\d{4})_(\d{2})_(\d{2})'  # YYYY_MM_DD
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                date_str = match.group(0)
                try:
                    # Handle different formats
                    if len(date_str) == 8 and date_str.isdigit():  # YYYYMMDD
                        return datetime.strptime(date_str, '%Y%m%d')
                    elif '-' in date_str:
                        if date_str[2] == '-':  # DD-MM-YYYY
                            return datetime.strptime(date_str, '%d-%m-%Y')
                        else:  # YYYY-MM-DD
                            return datetime.strptime(date_str, '%Y-%m-%d')
                    elif '_' in date_str:  # YYYY_MM_DD
                        year, month, day = date_str.split('_')
                        return datetime(int(year), int(month), int(day))
                except ValueError:
                    continue
        
        return None
    
    def should_skip_file(filename, folder_path, cutoff_date):
        """
        Determine if a file should be skipped based on date in filename or folder path.
        Returns True if file should be skipped, False otherwise.
        """
        if cutoff_date is None:
            return False
            
        # Try to extract date from filename
        file_date = extract_date_from_string(filename)
        if file_date is not None and file_date > cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0):
            return True
            
        # Try to extract date from folder path
        folder_date = extract_date_from_string(folder_path)
        if folder_date is not None and folder_date > cutoff_date.replace(hour=0, minute=0, second=0, microsecond=0):
            return True
            
        return False
    
    def process_files_in_dir(dir_path):
        files_processed = 0
        files_skipped = 0
        
        # List all files in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            # If recursive is enabled and we find a directory, process it
            if recursive and os.path.isdir(file_path):
                proc, skip = process_files_in_dir(file_path)
                files_processed += proc
                files_skipped += skip
                continue
                
            # Skip non-CSV files
            if not filename.endswith('.csv'):
                continue
                
            # Check if the file matches the pattern
            if file_pattern and file_pattern not in filename:
                continue
                
            # Check if the file matches the date filter
            if date_filter and date_filter not in filename:
                continue
            
            # Check if we should skip this file based on date in filename or folder path
            if should_skip_file(filename, dir_path, cutoff_date):
                print(f"Skipping file {filename} from {dir_path} - date after cutoff {cutoff_date.date()}")
                skipped_files.append(f"{dir_path}/{filename}")
                files_skipped += 1
                continue
            
            # Load the file (since it passes all our filters)
            try:
                df = pd.read_csv(file_path)
                
                # Skip empty files
                if df.empty:
                    print(f"Skipping empty file: {filename}")
                    continue
                
                # Add source filename as a column for reference
                df['source_file'] = filename
                # Add directory path for debugging
                df['source_dir'] = dir_path
                
                all_data.append(df)
                files_processed += 1
                print(f"Loaded {filename} from {dir_path}, {len(df)} rows")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return files_processed, files_skipped
    
    # Process the main directory (and subdirectories if recursive=True)
    files_found, files_skipped = process_files_in_dir(directory)
    
    print(f"Files processed: {files_found}")
    print(f"Files skipped due to date filtering: {files_skipped}")
    
    if files_found == 0:
        print(f"No matching files found in {directory} {'and subdirectories' if recursive else ''}.")
        if files_skipped > 0:
            print(f"Note: {files_skipped} files were skipped due to date filtering.")
        return pd.DataFrame()
    
    # Combine all dataframes
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df


def prepare_pv_data(pv_df, target_variable='pac'):
    """
    Prepare PV production data with focus on inverter data from min files.
    
    Parameters:
        pv_df (DataFrame): Raw PV data (min files)
        target_variable (str): Variable to predict - 'pac' for instant power or 'daysum' for daily energy
        
    Returns:
        DataFrame: Processed PV data
    """
    # Create a copy to avoid modifying the original
    df = pv_df.copy()
    
    # Check if the dataframe is empty
    if df.empty:
        print("Warning: Empty PV dataframe.")
        return df
    
    # Standardize column names (lowercase)
    df.columns = [col.lower() for col in df.columns]
    
    # Check if this is a min file with inverter data
    if 'pac' in df.columns and 'inv' in df.columns:
        print("Processing inverter data from min file")
        
        # Check for required columns
        required_cols = ['date', 'time', 'pac', 'inv']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing required columns in inverter data: {missing_cols}")
            return pd.DataFrame()
        
        # Create datetime column
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Extract time components for analysis
        df['hour'] = df['datetime'].dt.hour
        df['date_only'] = df['datetime'].dt.date
        
        # Different processing based on target variable
        if target_variable.lower() == 'pac':
            # For instant power (PAC), summarize by datetime
            print("Target: Instant power (PAC)")
            df_summarized = df.groupby('datetime').agg({
                'pac': 'sum',  # Sum power across all inverters
                'status': lambda x: (x == 6).sum()  # Count running inverters
            }).reset_index()
            
            # Rename target column
            df_summarized = df_summarized.rename(columns={'pac': 'pv_output'})
            
            # Convert to kW if needed (PAC is usually in W)
            if df_summarized['pv_output'].max() > 100:
                df_summarized['pv_output'] = df_summarized['pv_output'] / 1000
                print("Converting power from W to kW")
                
        elif target_variable.lower() == 'daysum':
            # For daily energy (DaySum), get the latest value per day
            print("Target: Daily energy sum (DaySum)")
            
            # Check if DaySum is available
            if 'daysum' not in df.columns:
                print("Warning: DaySum not found in data. Cannot predict daily energy.")
                return pd.DataFrame()
                
            # Get the latest reading for each day and inverter
            # Sort by datetime to ensure we get the latest value
            df = df.sort_values('datetime')
            
            # For each day and inverter, get the latest DaySum value
            latest_per_day_inv = df.groupby(['date_only', 'inv']).last().reset_index()
            
            # Sum across all inverters for each day
            df_summarized = latest_per_day_inv.groupby('date_only').agg({
                'daysum': 'sum'  # Total daily energy across all inverters
            }).reset_index()
            
            # Rename target column and datetime column
            df_summarized = df_summarized.rename(columns={
                'date_only': 'datetime',
                'daysum': 'pv_output'
            })
            
            # Convert to kWh if needed (DaySum is usually in Wh)
            if df_summarized['pv_output'].max() > 1000:
                df_summarized['pv_output'] = df_summarized['pv_output'] / 1000
                print("Converting energy from Wh to kWh")
                
        else:
            print(f"Warning: Unknown target variable '{target_variable}'. Using PAC as default.")
            # Default to PAC processing
            df_summarized = df.groupby('datetime').agg({
                'pac': 'sum'
            }).reset_index()
            df_summarized = df_summarized.rename(columns={'pac': 'pv_output'})
            
            if df_summarized['pv_output'].max() > 100:
                df_summarized['pv_output'] = df_summarized['pv_output'] / 1000
                
        # Add hour column if needed for further processing
        if 'hour' not in df_summarized.columns and 'datetime' in df_summarized.columns:
            df_summarized['hour'] = df_summarized['datetime'].dt.hour
            
        # Set datetime as index
        df_summarized = df_summarized.set_index('datetime').sort_index()
        
        return df_summarized
    else:
        print("Warning: Input data does not appear to be inverter data (missing PAC or INV columns).")
        return pd.DataFrame()
    

    


def prepare_weather_data(temp_df, humidity_df=None):
    """
    Prepare and combine temperature and humidity data.
    
    Parameters:
        temp_df (DataFrame): Temperature data
        humidity_df (DataFrame, optional): Humidity data
        
    Returns:
        DataFrame: Combined weather data
    """
    # Process temperature data
    if not temp_df.empty:
        temp_df = temp_df.copy()
        temp_df.columns = [col.lower() for col in temp_df.columns]
        
        if all(col in temp_df.columns for col in ['date', 'time', 'value']):
            temp_df['datetime'] = pd.to_datetime(temp_df['date'] + ' ' + temp_df['time'])
            temp_df = temp_df.rename(columns={'value': 'temperature'})
            temp_df = temp_df.drop_duplicates(subset=['datetime'])
            temp_df = temp_df.set_index('datetime').sort_index()
        else:
            print("Warning: Temperature data missing required columns.")
            return pd.DataFrame()
    else:
        print("Warning: Empty temperature dataframe.")
        return pd.DataFrame()
    
    # If humidity data is provided, process and merge it
    if humidity_df is not None and not humidity_df.empty:
        humidity_df = humidity_df.copy()
        humidity_df.columns = [col.lower() for col in humidity_df.columns]
        
        if all(col in humidity_df.columns for col in ['date', 'time', 'value']):
            humidity_df['datetime'] = pd.to_datetime(humidity_df['date'] + ' ' + humidity_df['time'])
            humidity_df = humidity_df.rename(columns={'value': 'humidity'})
            humidity_df = humidity_df.drop_duplicates(subset=['datetime'])
            humidity_df = humidity_df.set_index('datetime').sort_index()
            
            # Merge temperature and humidity
            weather_df = pd.merge(
                temp_df[['temperature']],
                humidity_df[['humidity']],
                how='outer',
                left_index=True,
                right_index=True
            )
        else:
            print("Warning: Humidity data missing required columns.")
            weather_df = temp_df[['temperature']]
    else:
        # Use only temperature data
        weather_df = temp_df[['temperature']]
    
    return weather_df


def filter_prediction_data(pred_df, cutoff_date=None):
    """
    Prepare and filter weather prediction data.
    
    Parameters:
        pred_df (DataFrame): Raw prediction data
        cutoff_date (datetime, optional): Cutoff date for training data
        
    Returns:
        DataFrame: Processed prediction data
    """
    if pred_df.empty:
        print("Warning: Empty prediction dataframe.")
        return pd.DataFrame()
    
    df = pred_df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Check required columns
    required_cols = ['date', 'time', 'pred_glob_ctrl', 'pred_t_2m_ctrl']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Prediction data missing required columns: {[col for col in required_cols if col not in df.columns]}")
        return pd.DataFrame()
    
    # Create datetime and target datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    # Handle prediction horizon
    if 'prediction' in df.columns:
        df['target_datetime'] = df['datetime'] + pd.to_timedelta(df['prediction'], unit='h')
    else:
        # If no prediction column, assume it's current data
        df['target_datetime'] = df['datetime']
        df['prediction'] = 0
    
    # Extract hour for easier analysis
    df['hour'] = df['target_datetime'].dt.hour
    
    # Apply cutoff date filtering
    if cutoff_date is not None:
        print(f"Filtering prediction data with cutoff date: {cutoff_date}")
        
        # IMPORTANT: Filter by target_datetime (when the prediction is for)
        # This prevents data leakage by ensuring we don't train on future weather data
        original_count = len(df)
        df = df[df['target_datetime'] <= cutoff_date]
        filtered_count = len(df)
        
        print(f"Filtered out {original_count - filtered_count} rows beyond cutoff date")
    
    # Set target_datetime as index
    df = df.set_index('target_datetime').sort_index()
    
    return df


def merge_datasets(pv_df, weather_df=None, pred_df=None, tolerance='1h'):
    """
    Merge PV, weather, and prediction data with different time granularities.
    - PV data: 5-minute intervals
    - Weather predictions: hourly intervals
    
    Parameters:
        pv_df (DataFrame): Processed PV data (5-minute intervals)
        weather_df (DataFrame, optional): Processed weather data
        pred_df (DataFrame, optional): Processed prediction data (hourly intervals)
        tolerance (str): Time tolerance for merging asof (default: 1h to match hourly predictions)
        
    Returns:
        DataFrame: Merged dataset
    """
    import pandas as pd
    import numpy as np
    
    # Check if PV dataframe is empty
    if pv_df.empty:
        print("Warning: Cannot merge with empty PV dataframe.")
        return pd.DataFrame()
    
    # Reset indices for merging
    pv_reset = pv_df.reset_index()
    
    # Make sure pv_reset has a datetime column
    if 'datetime' not in pv_reset.columns:
        print("Error: PV data does not have a datetime column.")
        return pd.DataFrame()
    
    # Round the 5-minute timestamps to the nearest hour to match prediction data granularity
    pv_reset['datetime_hourly'] = pv_reset['datetime'].dt.floor('H')
    
    # Initialize merged dataframe with PV data
    merged = pv_reset.copy()
    
    # Add time-based features
    merged['hour_sin'] = np.sin(2 * np.pi * merged['datetime'].dt.hour / 24)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['datetime'].dt.hour / 24)
    merged['day_of_year'] = merged['datetime'].dt.dayofyear / 365.0
    
    # If prediction data is provided and not empty, merge it
    if pred_df is not None and not pred_df.empty:
        pred_reset = pred_df.reset_index()
        
        # Check which datetime column to use from pred_df
        pred_datetime_col = None
        if 'target_datetime' in pred_reset.columns:
            pred_datetime_col = 'target_datetime'
        elif 'datetime' in pred_reset.columns:
            pred_datetime_col = 'datetime'
        
        if pred_datetime_col:
            # Round prediction timestamps to the hour to ensure consistency
            pred_reset['datetime_hourly'] = pred_reset[pred_datetime_col].dt.floor('H')
            
            # Create a simple dataframe with only the necessary prediction columns
            pred_columns = [col for col in pred_reset.columns if col.startswith('pred_')]
            
            if pred_columns:
                # Create a dataframe with hourly prediction data
                hourly_pred_df = pred_reset[['datetime_hourly'] + pred_columns].copy()
                
                # If there are duplicate datetime_hourly entries (e.g., multiple predictions for same hour),
                # keep the most recent one (assuming sorted by datetime)
                hourly_pred_df = hourly_pred_df.drop_duplicates(subset=['datetime_hourly'], keep='last')
                
                # Merge PV and prediction data based on the hourly datetime
                try:
                    merged = pd.merge(
                        merged,
                        hourly_pred_df,
                        on='datetime_hourly',
                        how='left'
                    )
                    print(f"Successfully merged prediction data with {len(pred_columns)} prediction columns.")
                except Exception as e:
                    print(f"Error merging prediction data: {str(e)}")
                    # Continue with just the PV data
            else:
                print("Warning: No prediction columns found in prediction data.")
        else:
            print("Warning: Prediction data does not have a suitable datetime column, skipping prediction merge.")
    
    # Filter for daylight hours (simplistic approach - adjust as needed)
    if 'hour' not in merged.columns:
        merged['hour'] = merged['datetime'].dt.hour
        
    daylight_df = merged[(merged['hour'] >= 6) & (merged['hour'] <= 19)]
    
    if daylight_df.empty:
        print("Warning: No data points during daylight hours (6-19).")
        return merged  # Return all data instead of empty dataframe
    
    # Drop the temporary datetime_hourly column if it exists
    if 'datetime_hourly' in daylight_df.columns:
        daylight_df = daylight_df.drop(columns=['datetime_hourly'])
    
    # Check if we have any prediction columns in the merged data
    pred_cols_in_merged = [col for col in daylight_df.columns if col.startswith('pred_')]
    if not pred_cols_in_merged:
        print("Warning: No prediction columns available in merged dataset. Check data compatibility.")
    else:
        print(f"Merged dataset contains {len(pred_cols_in_merged)} prediction columns: {pred_cols_in_merged}")
    
    return daylight_df


def handle_missing_data(df, method='interpolate'):
    """
    Handle missing data in the dataset.
    
    Parameters:
        df (DataFrame): Input dataframe with potential missing values
        method (str): Method to handle missing values ('interpolate', 'drop', or 'fill')
        
    Returns:
        DataFrame: Dataframe with handled missing values
    """
    if df.empty:
        return df
    
    # Check for missing values
    missing_count = df.isna().sum()
    if missing_count.sum() > 0:
        print("Missing values before handling:")
        print(missing_count[missing_count > 0])
    
    processed_df = df.copy()
    
    # Select only numeric columns for interpolation
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle non-numeric columns separately
    object_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = [col for col in processed_df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
    
    # Remove datetime columns from interpolation
    for col in datetime_cols:
        if col in numeric_cols:
            numeric_cols.remove(col)
    
    if method == 'interpolate':
        # Interpolate missing values for numeric columns only
        for col in numeric_cols:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].interpolate(method='linear').ffill().bfill()
        
        # Forward fill and backward fill for object columns
        for col in object_cols:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].ffill().bfill()
        
    elif method == 'drop':
        # Drop rows with any missing values in numeric columns
        processed_df = processed_df.dropna(subset=numeric_cols)
        
    elif method == 'fill':
        # Fill missing values with appropriate defaults
        # Fill prediction data with zeros if missing
        for col in numeric_cols:
            if col.startswith('pred_') and processed_df[col].isna().any():
                processed_df[col] = processed_df[col].fillna(0)
        
        # Fill temperature with the daily mean
        if 'temperature' in numeric_cols and processed_df['temperature'].isna().any():
            # Group by date and fill with daily mean
            if 'date' not in processed_df.columns and 'datetime' in processed_df.columns:
                processed_df['date'] = processed_df['datetime'].dt.date
                
            if 'date' in processed_df.columns:
                daily_mean_temp = processed_df.groupby('date')['temperature'].transform('mean')
                processed_df['temperature'] = processed_df['temperature'].fillna(daily_mean_temp)
        
        # Fill humidity with the daily mean
        if 'humidity' in numeric_cols and processed_df['humidity'].isna().any():
            if 'date' not in processed_df.columns and 'datetime' in processed_df.columns:
                processed_df['date'] = processed_df['datetime'].dt.date
                
            if 'date' in processed_df.columns:
                daily_mean_humidity = processed_df.groupby('date')['humidity'].transform('mean')
                processed_df['humidity'] = processed_df['humidity'].fillna(daily_mean_humidity)
        
        # Fill remaining missing numeric values
        for col in numeric_cols:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].interpolate(method='linear').ffill().bfill()
        
        # Fill missing object values
        for col in object_cols:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].ffill().bfill()
    
    # Check if there are still missing values
    still_missing = processed_df.isna().sum()
    if still_missing.sum() > 0:
        print("Missing values after handling:")
        print(still_missing[still_missing > 0])
    
    return processed_df


def prepare_model_features_predictions_only(df):
    """
    Prepare features for model training using only weather predictions (no measurements).
    """
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    # Select only prediction features
    features = []
    
    # Solar radiation prediction (most important)
    if 'pred_glob_ctrl' in df.columns:
        features.append('pred_glob_ctrl')
    
    # Temperature prediction
    if 'pred_t_2m_ctrl' in df.columns:
        features.append('pred_t_2m_ctrl')
    
    # Humidity prediction
    if 'pred_relhum_2m_ctrl' in df.columns:
        features.append('pred_relhum_2m_ctrl')
    
    # Precipitation prediction (if available)
    if 'pred_tot_prec_ctrl' in df.columns:
        features.append('pred_tot_prec_ctrl')
    
    # Time-based features (always available)
    if 'hour_sin' in df.columns and 'hour_cos' in df.columns:
        features.extend(['hour_sin', 'hour_cos'])
    if 'day_of_year' in df.columns:
        features.append('day_of_year')
    
    # Create feature matrix and target vector
    X = df[features].copy()
    
    # Target variable
    if 'pv_output' in df.columns:
        y = df['pv_output']
    else:
        print("Warning: Target variable 'pv_output' not found.")
        return X, None
    
    return X, y


def prepare_model_features(df):
    """
    Prepare features for model training.
    
    Parameters:
        df (DataFrame): Input dataframe with raw data
        
    Returns:
        tuple: X (features) and y (target)
    """
    if df.empty:
        return pd.DataFrame(), pd.Series()
    
    # Select relevant features
    features = []
    
    # Solar radiation is the most important predictor (if available)
    if 'pred_glob_ctrl' in df.columns:
        features.append('pred_glob_ctrl')
    # Temperature features
    if 'pred_t_2m_ctrl' in df.columns:
        features.append('pred_t_2m_ctrl')
    if 'temperature' in df.columns:
        features.append('temperature')
    
    # Humidity features
    if 'pred_relhum_2m_ctrl' in df.columns:
        features.append('pred_relhum_2m_ctrl')
    if 'humidity' in df.columns:
        features.append('humidity')
    
    # Time-based features
    if 'hour_sin' in df.columns and 'hour_cos' in df.columns:
        features.extend(['hour_sin', 'hour_cos'])
    if 'day_of_year' in df.columns:
        features.append('day_of_year')
    
    # Check if we have enough features
    if len(features) < 2:
        print("Warning: Not enough features available for modeling.")
        if df.shape[1] > 1:
            # Use all available numeric columns except the target
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'pv_output' in features:
                features.remove('pv_output')
        else:
            return pd.DataFrame(), pd.Series()
    
    # Create feature matrix and target vector
    X = df[features].copy()
    
    # Target variable should be 'pv_output'
    if 'pv_output' in df.columns:
        y = df['pv_output']
    else:
        print("Warning: Target variable 'pv_output' not found.")
        return X, None
    
    return X, y

# ---------------------------------
# Model Training and Evaluation
# ---------------------------------

def train_model(X, y, model_type='random_forest'):
    """
    Train a prediction model.
    
    Parameters:
        X (DataFrame): Feature matrix
        y (Series): Target variable
        model_type (str): Type of model to train ('random_forest', 'gbm', or 'linear')
        
    Returns:
        dict: Trained model and evaluation metrics
    """
    if X.empty or y is None or y.empty:
        print("Error: Cannot train model with empty data.")
        return None
    
    print(f"Training {model_type} model with {X.shape[1]} features and {X.shape[0]} samples")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (important for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model based on specified type
    if model_type == 'random_forest':
        # Random Forest model (good for capturing non-linear relationships)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Parameter grid for optimization
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
        
    elif model_type == 'gbm':
        # Gradient Boosting model (often performs well for time series)
        model = GradientBoostingRegressor(random_state=42)
        
        # Parameter grid for optimization
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
        
    else:  # Default to linear regression
        # Linear model (simpler, but may miss non-linear relationships)
        model = LinearRegression()
        # Linear regression has no hyperparameters to tune
        param_grid = {}
    
    # Perform hyperparameter tuning if applicable
    if param_grid:
        print("Performing hyperparameter tuning...")
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_
        print(f"Best parameters: {grid_search.best_params_}")
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate evaluation metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    print(f"Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # If using tree-based model, print feature importance
    if model_type in ['random_forest', 'gbm']:
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nFeature Importance:")
        print(feature_importance)
    
    # Return trained model, scaler, and evaluation metrics
    return {
        'model': best_model,
        'scaler': scaler,
        'metrics': metrics,
        'features': X.columns.tolist(),
        'feature_importance': feature_importance if model_type in ['random_forest', 'gbm'] else None
    }


def predict_pv_output(model_info, prediction_data):
    """
    Predict PV output using the trained model.
    
    Parameters:
        model_info (dict): Dictionary containing model, scaler, and feature list
        prediction_data (DataFrame): Weather prediction data with the same features used during training
        
    Returns:
        DataFrame: Original data with added PV output predictions
    """
    if model_info is None:
        print("Error: No model provided for prediction.")
        return prediction_data
    
    # Extract model components
    model = model_info['model']
    scaler = model_info['scaler']
    features = model_info['features']
    
    # Check if prediction data is empty
    if prediction_data.empty:
        print("Error: Empty prediction data.")
        return prediction_data
    
    # Ensure all required features are present
    missing_features = [feature for feature in features if feature not in prediction_data.columns]
    if missing_features:
        print(f"Warning: Missing features in prediction data: {missing_features}")
        print("Attempting to create or approximate missing features...")
        
        # Try to create missing time-based features
        if 'hour_sin' in missing_features and 'datetime' in prediction_data.columns:
            prediction_data['hour_sin'] = np.sin(2 * np.pi * prediction_data['datetime'].dt.hour / 24)
        
        if 'hour_cos' in missing_features and 'datetime' in prediction_data.columns:
            prediction_data['hour_cos'] = np.cos(2 * np.pi * prediction_data['datetime'].dt.hour / 24)
        
        if 'day_of_year' in missing_features and 'datetime' in prediction_data.columns:
            prediction_data['day_of_year'] = prediction_data['datetime'].dt.dayofyear / 365.0
        
        # For remaining missing features, we'll have to skip them
        still_missing = [feature for feature in features if feature not in prediction_data.columns]
        if still_missing:
            print(f"Cannot create these features: {still_missing}")
            print("Proceeding with available features only.")
            # Use only available features
            features = [feature for feature in features if feature in prediction_data.columns]
    
    # Extract features used by the model
    X_pred = prediction_data[features].copy()
    
    # Scale features
    X_pred_scaled = scaler.transform(X_pred)
    
    # Make predictions
    predictions = model.predict(X_pred_scaled)
    
    # Add predictions to the dataframe
    prediction_data['predicted_pv_output'] = predictions
    
    # Ensure non-negative predictions (PV output can't be negative)
    prediction_data['predicted_pv_output'] = prediction_data['predicted_pv_output'].clip(lower=0)
    
    return prediction_data


def get_cross_validation_scores(model_info, X, y, cv=5):
    """
    Calculate cross-validation scores for a trained model.
    
    Parameters:
        model_info (dict): Dictionary containing model and related information
        X (DataFrame): Feature matrix
        y (Series): Target variable
        cv (int): Number of folds for cross-validation
        
    Returns:
        dict: Dictionary of cross-validation scores
    """
    
    model = model_info['model']
    scaler = model_info['scaler']
    X_scaled = scaler.transform(X)
    
    # Calculate different CV scores
    cv_r2_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
    cv_neg_mse_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_squared_error')
    cv_neg_mae_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='neg_mean_absolute_error')
    
    return {
        'R²': cv_r2_scores,
        'MSE': -cv_neg_mse_scores,
        'MAE': -cv_neg_mae_scores,
        'RMSE': np.sqrt(-cv_neg_mse_scores)
    }

def visualize_model_results(model_info, X, y, model_name="Model", cv_scores=None, num_folds=5):
    """
    Create a comprehensive visualization of model results including:
    - Residual plot 
    - Actual vs Predicted values
    - Model evaluation metrics (MAE, MSE, RMSE, R², RSS)
    - Cross-validation scores
    
    Parameters:
        model_info (dict): Dictionary containing model and related information
        X (DataFrame): Feature matrix used for prediction
        y (Series): Actual target values
        model_name (str): Name of the model to display
        cv_scores (dict, optional): Pre-computed cross-validation scores
        num_folds (int): Number of folds for cross-validation if cv_scores not provided
        
    Returns:
        fig: The matplotlib figure object
    """
    
    if model_info is None:
        print("Error: No model provided.")
        return None
    
    # Extract model and scaler
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions
    y_pred = model.predict(X_scaled)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    rss = np.sum(residuals**2)  # Residual Sum of Squares
    
    # Compute cross-validation scores if not provided
    if cv_scores is None:
        cv_r2_scores = cross_val_score(model, X_scaled, y, cv=num_folds, scoring='r2')
        cv_neg_mse_scores = cross_val_score(model, X_scaled, y, cv=num_folds, scoring='neg_mean_squared_error')
        cv_neg_mae_scores = cross_val_score(model, X_scaled, y, cv=num_folds, scoring='neg_mean_absolute_error')
        
        cv_scores = {
            'R²': cv_r2_scores,
            'MSE': -cv_neg_mse_scores,
            'MAE': -cv_neg_mae_scores,
            'RMSE': np.sqrt(-cv_neg_mse_scores)
        }
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    
    # 1. Residual plot (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_pred, residuals, alpha=0.5, color='blue')
    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residual Plot')
    ax1.grid(True, alpha=0.3)
    
    # Add text with model type 
    ax1.text(0.05, 0.95, f"Model: {model_name}", transform=ax1.transAxes, 
             fontsize=12, fontweight='bold', verticalalignment='top')
    
    # 2. Actual vs Predicted plot (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(y, y_pred, alpha=0.5, color='green')
    max_val = max(y.max(), y_pred.max())
    min_val = min(y.min(), y_pred.min())
    buffer = (max_val - min_val) * 0.1
    ax2.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'r--')
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Predicted Values')
    ax2.set_title('Actual vs Predicted')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error distribution histogram (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    sns.histplot(residuals, kde=True, ax=ax3, color='purple')
    ax3.axvline(x=0, color='r', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Residual Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution')
    
    # 4. Metrics table (bottom-left)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.axis('off')
    metrics_text = f"""
    MODEL EVALUATION METRICS
    ========================
    MAE:  {mae:.4f}
    MSE:  {mse:.4f}
    RMSE: {rmse:.4f}
    R²:   {r2:.4f}
    RSS:  {rss:.4f}
    """
    ax4.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', verticalalignment='center')
    
    # 5. Cross-validation results (bottom-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    cv_means = [np.mean(cv_scores[metric]) for metric in ['R²', 'MAE', 'MSE', 'RMSE']]
    cv_stds = [np.std(cv_scores[metric]) for metric in ['R²', 'MAE', 'MSE', 'RMSE']]
    cv_metrics = ['R²', 'MAE', 'MSE', 'RMSE']
    
    colors = ['green', 'blue', 'orange', 'red']
    bar_positions = range(len(cv_metrics))
    
    bars = ax5.bar(bar_positions, cv_means, yerr=cv_stds, alpha=0.7, color=colors, capsize=10)
    
    # Add values on top of bars
    for bar, value, std in zip(bars, cv_means, cv_stds):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9, rotation=0)
    
    ax5.set_ylabel('Score')
    ax5.set_title(f'{num_folds}-Fold Cross-Validation Scores')
    ax5.set_xticks(bar_positions)
    ax5.set_xticklabels(cv_metrics)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Feature importance (bottom-right)
    ax6 = fig.add_subplot(gs[1, 2])
    
    if 'feature_importance' in model_info and model_info['feature_importance'] is not None:
        # Use model's feature importance if available
        importance_df = model_info['feature_importance']
        
        # Limit to top 10 features if there are many
        if len(importance_df) > 10:
            importance_df = importance_df.head(10)
            
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax6)
        ax6.set_title('Top Feature Importance')
    else:
        # For models without built-in feature importance
        ax6.text(0.5, 0.5, "Feature importance not available\nfor this model type", 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Feature Importance')
    
    # Add overall title
    fig.suptitle(f'Comprehensive Evaluation of {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    return fig


def visualize_predictions(actual, predicted, title="Model Performance"):
    """
    Visualize model predictions against actual values.
    
    Parameters:
        actual (Series): Actual PV output values
        predicted (Series): Predicted PV output values
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Create scatter plot
    plt.scatter(actual, predicted, alpha=0.5)
    
    # Add perfect prediction line
    max_val = max(actual.max(), predicted.max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    
    plt.xlabel('Actual PV Output')
    plt.ylabel('Predicted PV Output')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_feature_importance(model_info):
    """
    Visualize feature importance for tree-based models.
    
    Parameters:
        model_info (dict): Dictionary containing model and feature information
    """
    if model_info is None or 'feature_importance' not in model_info or model_info['feature_importance'] is None:
        print("Feature importance visualization not available for this model.")
        return
    
    importance_df = model_info['feature_importance']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

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


def plot_daily_production(df, date_col='date', value_col='predicted_pv_output'):
    """
    Plot daily production forecasts.
    
    Parameters:
        df (DataFrame): DataFrame containing production data
        date_col (str): Column name for dates
        value_col (str): Column name for production values
    """
    # Check if datetime column exists
    if 'datetime' in df.columns and date_col not in df.columns:
        df[date_col] = df['datetime'].dt.date
    
    # Group by date and sum production
    daily_sum = df.groupby(date_col)[value_col].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.bar(daily_sum[date_col].astype(str), daily_sum[value_col], color='skyblue')
    plt.xlabel('Date')
    plt.ylabel('Total Daily Production')
    plt.title('Forecasted Daily PV Production')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_predictions_to_csv(predictions_df, output_path=None, include_date=True):
        """
        Save prediction results to a CSV file.
        
        Parameters:
            predictions_df (DataFrame): DataFrame with predictions
            output_path (str, optional): Path to save the CSV file
            include_date (bool): Whether to include current date in filename
            
        Returns:
            str: Path to the saved CSV file
        """
        if predictions_df is None or predictions_df.empty:
            print("Error: No predictions to save.")
            return None
        
        # Create default output directory if needed
        if output_path is None:
            output_dir = 'predictions'
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with current date and time
            if include_date:
                current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f"pv_predictions_{current_date}.csv"
            else:
                filename = "pv_predictions.csv"
                
            output_path = os.path.join(output_dir, filename)
        
        # Prepare data for output
        output_df = predictions_df.copy()
        
        # Convert datetime index to column if it's an index
        if isinstance(output_df.index, pd.DatetimeIndex):
            output_df = output_df.reset_index()
        
        # Make sure datetime is in a standard format
        if 'datetime' in output_df.columns:
            output_df['datetime'] = output_df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to CSV
        try:
            output_df.to_csv(output_path, index=False)
            print(f"Predictions saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving predictions: {e}")
            return None


def save_model(model_info, output_path=None):
        """
        Save the trained model and related information.
        
        Parameters:
            model_info (dict): Dictionary containing model and related information
            output_path (str, optional): Path to save the model
            
        Returns:
            str: Path to the saved model
        """
        if model_info is None:
            print("Error: No model to save.")
            return None
        
        # Create default output directory if needed
        if output_path is None:
            output_dir = 'models'
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename with current date
            current_date = datetime.now().strftime('%Y-%m-%d')
            output_path = os.path.join(output_dir, f"pv_prediction_model_{current_date}.pkl")
        
        # Save the model
        try:
            with open('models/prediction_model.pkl','wb') as f:
                pickle.dump(model_info, f)
            print(f"Model saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None
        

# ---------------------------------
# Main Execution Code
# ---------------------------------

def train_prediction_model(data_date=None, save_outputs=True, target_variable='pac', 
                           training_cutoff_days=3, visualization=False):
    """
    Main function to train a prediction-only model with temporal filtering to avoid data leakage.
    Uses ONLY inverter data (from SolarLogs) and weather prediction data (from Meteo).
    
    Parameters:
        data_date (str, optional): Date string in 'YYYY-MM-DD' format for data loading
        save_outputs (bool): Whether to save models and predictions to files
        target_variable (str): Target to predict - 'pac' for power or 'daysum' for daily energy
        training_cutoff_days (int): Number of days in the past to use as limit for training data
        
    Returns:
        dict: Dictionary containing model information and results
    """
    
    
    print("Solar Energy Production Prediction Model - Using ONLY Inverter and Weather Prediction Data")
    print("=" * 80)
    print(f"Target variable: {target_variable}")
    
    # Calculate cutoff date (typically a few days ago to avoid data leakage)
    cutoff_date = datetime.now() - timedelta(days=training_cutoff_days)
    print(f"Using training data cutoff date: {cutoff_date}")
    
    # Get all directories
    dirs = get_directories(data_date, use_all_folders=True)
    print(f"Using data directories:")
    for key, path in dirs.items():
        if key.startswith('all_'):
            print(f"  {key}: {len(path)} directories found")
        else:
            print(f"  {key}: {path}")
    
    def extract_date_from_path(path):
        """Extract date from path containing cleaned_data_YYYY-MM-DD"""
        match = re.search(r'cleaned_data_(\d{4}-\d{2}-\d{2})', path)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y-%m-%d')
            except ValueError:
                return None
        return None
    
    # Filter directories by cutoff date
    valid_data_dirs = []
    for data_dir in dirs['all_data_dirs']:
        dir_date = extract_date_from_path(data_dir)
        if dir_date and dir_date <= cutoff_date:
            valid_data_dirs.append(data_dir)
            print(f"Including directory: {data_dir}")
        else:
            print(f"Excluding directory (beyond cutoff date): {data_dir}")
    
    if not valid_data_dirs:
        print("WARNING: No data directories found before cutoff date. Using most recent available directory.")
        # Use the oldest available folder if no valid directories found
        dirs['all_data_dirs'].sort()  # Sort chronologically
        if dirs['all_data_dirs']:
            valid_data_dirs = [dirs['all_data_dirs'][0]]  # Take the oldest
            print(f"Including directory (fallback): {valid_data_dirs[0]}")
        else:
            print("ERROR: No data directories found at all.")
            return None
    
    # Load inverter data (min files) from all valid directories
    print("\nLoading inverter data...")
    all_valid_solar_dirs = [os.path.join(d, 'Solarlogs') for d in valid_data_dirs]
    
    inverter_df = pd.DataFrame()
    for solar_dir in all_valid_solar_dirs:
        if os.path.exists(solar_dir):
            df = load_filtered_csv_files(solar_dir, file_pattern='min', recursive=True)
            if not df.empty:
                print(f"Loaded {len(df)} inverter records from {solar_dir}")
                inverter_df = pd.concat([inverter_df, df]) if not inverter_df.empty else df
    
    if inverter_df.empty:
        print("No min files found. Trying PV files as fallback...")
        for solar_dir in all_valid_solar_dirs:
            if os.path.exists(solar_dir):
                df = load_filtered_csv_files(solar_dir, file_pattern='-PV', recursive=True)
                if not df.empty:
                    print(f"Loaded {len(df)} PV records from {solar_dir}")
                    inverter_df = pd.concat([inverter_df, df]) if not inverter_df.empty else df
    
    if inverter_df.empty:
        print("ERROR: No inverter or PV data found. Check file patterns and directory paths.")
        return None
    
    # Prepare PV data
    pv_data = prepare_pv_data(inverter_df, target_variable)
    if not pv_data.empty:
        print(f"PV data prepared: {pv_data.shape[0]} records")
    else:
        print("ERROR: Failed to prepare PV data. Check file formats and target variable.")
        return None
    
    # Skip temperature and humidity data loading - NOT USED

    # Empty dataframe for weather - we're only using prediction data
    weather_data = pd.DataFrame()
    
    # Load prediction data from all valid directories
    print("\nLoading prediction data...")
    all_valid_meteo_dirs = [os.path.join(d, 'Meteo') for d in valid_data_dirs]
    
    pred_df = pd.DataFrame()
    for meteo_dir in all_valid_meteo_dirs:
        if os.path.exists(meteo_dir):
            df = load_filtered_csv_files(meteo_dir, file_pattern='Pred', recursive=True)
            if not df.empty:
                print(f"Loaded {len(df)} prediction records from {meteo_dir}")
                pred_df = pd.concat([pred_df, df]) if not pred_df.empty else df
    
    # Apply special filtering for prediction data
    if not pred_df.empty:
        pred_data = filter_prediction_data(pred_df, cutoff_date=cutoff_date)
        print(f"Prediction data prepared: {pred_data.shape[0]} records")
    else:
        print("Warning: No prediction data available.")
        pred_data = pd.DataFrame()
    
    # For DaySum predictions, adjust data merging strategy
    if target_variable.lower() == 'daysum':
        # Need to adjust merge strategy for daily data
        print("Adapting merge strategy for daily predictions...")
        
        # Reset indices for merging
        pv_reset = pv_data.reset_index()
        
        # Skip weather data processing - NOT USED
        
        # Handle prediction data for daily averages
        if 'pred_data' in locals() and not pred_data.empty:
            pred_reset = pred_data.reset_index()
            if 'target_datetime' in pred_reset.columns:
                pred_reset['date'] = pred_reset['target_datetime'].dt.date
            elif 'datetime' in pred_reset.columns:
                pred_reset['date'] = pred_reset['datetime'].dt.date
            else:
                print("Warning: Prediction data has no datetime column. Cannot process for daily predictions.")
                pred_daily = pd.DataFrame()
            
            # For each prediction variable, get daily averages during daylight hours (6-19)
            if 'hour' in pred_reset.columns:
                daylight_mask = (pred_reset['hour'] >= 6) & (pred_reset['hour'] <= 19)
                # Get columns that are likely prediction variables
                pred_columns = [col for col in pred_reset.columns if col.startswith('pred_')]
                if pred_columns:
                    agg_dict = {}
                    for col in pred_columns:
                        if 'prec' in col.lower() or 'rain' in col.lower():
                            # Precipitation should be summed
                            agg_dict[col] = 'sum'
                        else:
                            # Temperature, humidity, radiation should be averaged
                            agg_dict[col] = 'mean'
                    
                    # Apply the aggregation
                    pred_daily = pred_reset[daylight_mask].groupby('date').agg(agg_dict).reset_index()
                    pred_daily['datetime'] = pd.to_datetime(pred_daily['date'])
                else:
                    print("Warning: No prediction variables found in prediction data.")
                    pred_daily = pd.DataFrame()
            else:
                print("Warning: Prediction data has no hour column. Cannot filter for daylight hours.")
                pred_daily = pd.DataFrame()
        
        # Merge datasets based on date instead of datetime
        merged_data = pv_reset
        
        # Skip weather_daily merging - NOT USED
            
        if 'pred_daily' in locals() and not pred_daily.empty:
            merged_data = pd.merge(
                merged_data,
                pred_daily,
                left_on='datetime',
                right_on='datetime',
                how='left'
            )
        
        # Add day of year feature
        if 'datetime' in merged_data.columns:
            merged_data['day_of_year'] = merged_data['datetime'].dt.dayofyear / 365.0
    else:
        # For PAC predictions, use the normal merge strategy
        print("\nMerging datasets...")
        merged_data = merge_datasets(pv_data, pd.DataFrame(), pred_data)  # Empty dataframe for weather
    
    if merged_data.empty:
        print("ERROR: No data available after merging. Check data compatibility.")
        return None
    
    print(f"Merged data: {merged_data.shape[0]} records, {merged_data.shape[1]} columns")
    print(f"Columns: {merged_data.columns.tolist()}")
    
    # Handle missing data
    print("\nHandling missing data...")
    clean_data = handle_missing_data(merged_data, method='interpolate')

    # Generate visualizations if there's enough data
    if visualization:
        if len(clean_data) > 10:  # Arbitrary threshold to avoid plotting with too little data
            print("\nGenerating correlation matrix...")
            try:
                corr_matrix = plot_correlation_matrix(clean_data, title=f"Feature Correlation Matrix for {target_variable.upper()} Prediction")
            except Exception as e:
                print(f"Warning: Could not generate correlation matrix: {str(e)}")
                
            print("\nGenerating feature pairplot...")
            try:
                # Limit sample size for large datasets
                sample_size = min(1000, len(clean_data))
                plot_feature_pairplot(clean_data, target_col='pv_output', sample_size=sample_size)
            except Exception as e:
                print(f"Warning: Could not generate feature pairplot: {str(e)}")
        else:
            print("Warning: Not enough data to generate visualizations.")
    
    # Prepare features for prediction-only model
    print("\nPreparing prediction-only model features...")
    X_pred_only, y_same = prepare_model_features_predictions_only(clean_data)
    
    results = {}
    
    # Train and evaluate prediction-only model
    if not X_pred_only.empty and y_same is not None and len(X_pred_only) > 10:
        print("\nTraining prediction-only model...")
        model_info = train_model(X_pred_only, y_same, model_type='random_forest')
        predictions = predict_pv_output(model_info, clean_data)
        
        print("\nModel metrics:")
        print(f"RMSE: {model_info['metrics']['RMSE']:.4f}")
        print(f"MAE: {model_info['metrics']['MAE']:.4f}")
        print(f"R2: {model_info['metrics']['R2']:.4f}")
        
        results['model'] = model_info
        results['predictions'] = predictions
        
        # Print model feature importance
        if 'feature_importance' in model_info and model_info['feature_importance'] is not None:
            print("\nModel feature importance:")
            print(model_info['feature_importance'])
            
        # Calculate cross-validation scores for the model
        print("\nCalculating cross-validation scores...")
        cv_scores = get_cross_validation_scores(model_info, X_pred_only, y_same, cv=5)

        if visualization:
            # Generate comprehensive model evaluation visualization
            print("\nGenerating comprehensive evaluation visualization...")
            try:
                fig = visualize_model_results(
                    model_info, 
                    X_pred_only, 
                    y_same, 
                    model_name="Weather Prediction Model",
                    cv_scores=cv_scores
                )
            
                # Save model evaluation visualization if requested
                if save_outputs:
                    figures_dir = 'figures'
                    os.makedirs(figures_dir, exist_ok=True)
                    fig_path = os.path.join(figures_dir, f"model_evaluation_{target_variable}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
                    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    print(f"Model evaluation visualization saved to: {fig_path}")
            except Exception as e:
                print(f"Warning: Could not generate model evaluation visualization: {str(e)}")
            
    else:
        print(f"WARNING: Cannot train model due to insufficient data (features: {X_pred_only.shape if not X_pred_only.empty else 'empty'}, target: {len(y_same) if y_same is not None else 'None'}).")
        return None
    
    # Save outputs if requested
    if save_outputs:
        # Create output directories
        output_dir = 'predictions'
        model_dir = 'models'
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create filenames with target variable and current date
        target_suffix = target_variable.lower()
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Save model
        model_path = os.path.join(model_dir, f"prediction_model.pkl")
        save_model(results['model'], output_path=model_path)
        print(f"Model saved to: {model_path}")
        
        # Save predictions
        prediction_path = os.path.join(output_dir, f"predictions_{target_suffix}_{current_date}.csv")
        save_predictions_to_csv(results['predictions'], output_path=prediction_path)
        print(f"Predictions saved to: {prediction_path}")
        
        # Save the path for the pipeline function to return
        results['prediction_path'] = prediction_path
    
    # Generate detailed report
    print("\n" + "="*60)
    print("MODEL SUMMARY REPORT")
    print("="*60)
    
    # Calculate RSS
    residuals = clean_data['pv_output'] - results['predictions']['predicted_pv_output']
    rss = np.sum(residuals**2)
    
    print(f"Model Type: {target_variable.upper()} Prediction")
    print(f"Features: {', '.join(results['model']['features'])}")
    print("\nPerformance Metrics:")
    print(f"  RMSE: {results['model']['metrics']['RMSE']:.4f}")
    print(f"  MAE: {results['model']['metrics']['MAE']:.4f}")
    print(f"  R²: {results['model']['metrics']['R2']:.4f}")
    print(f"  RSS: {rss:.4f}")
    
    # Create and save summary report as text file
    if save_outputs:
        report_dir = 'reports'
        os.makedirs(report_dir, exist_ok=True)
        
        report_path = os.path.join(report_dir, f"model_report_{target_variable}_{current_date}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SOLAR ENERGY PRODUCTION PREDICTION MODEL REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Target Variable: {target_variable.upper()}\n")
            f.write(f"Data Date: {data_date if data_date else 'Not specified'}\n")
            f.write(f"Training Cutoff Date: {cutoff_date}\n\n")
            
            f.write("MODEL FEATURES\n")
            f.write("-" * 40 + "\n")
            f.write(f"{', '.join(results['model']['features'])}\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Metric':<15} {'Value':<15}\n")
            f.write("-" * 30 + "\n")
            
            # Metrics
            rmse = results['model']['metrics']['RMSE']
            mae = results['model']['metrics']['MAE']
            r2 = results['model']['metrics']['R2']
            f.write(f"{'RMSE':<15} {rmse:<15.4f}\n")
            f.write(f"{'MAE':<15} {mae:<15.4f}\n")
            f.write(f"{'R²':<15} {r2:<15.4f}\n")
            f.write(f"{'RSS':<15} {rss:<15.4f}\n\n")
            
            # CV Scores if available
            if 'cv_scores' in locals():
                f.write("CROSS-VALIDATION SCORES (MEAN ± STD)\n")
                f.write("-" * 40 + "\n")
                
                metrics = ['R²', 'MAE', 'MSE', 'RMSE']
                
                for metric in metrics:
                    mean = np.mean(cv_scores[metric])
                    std = np.std(cv_scores[metric])
                    f.write(f"{metric:<10}: {mean:.4f} ± {std:.4f}\n")
            
            f.write("\n\nThis report was generated automatically by the Solar Energy Production Prediction Model.")
            f.write(f"\nTraining data was filtered to exclude data after {cutoff_date} to prevent data leakage.")
        
        print(f"Detailed report saved to: {report_path}")
            
    return results

def solar_prediction_pipeline(data_date=None, save_outputs=True, target_variable='pac', 
                             training_cutoff_days=3):
    """
    Complete pipeline for solar production prediction:
    1. Trains a prediction-only model with temporal filtering to avoid data contamination
    
    Parameters:
        data_date (str, optional): Date string for loading data
        save_outputs (bool): Save outputs
        target_variable (str): 'pac' for power or 'daysum' for energy
        training_cutoff_days (int): Days to exclude from training
        
    Returns:
        str: Path to the prediction outputs or None if failed
    """
    
    print("=" * 80)
    print("COMPLETE SOLAR PRODUCTION PREDICTION PIPELINE")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get directories
    dirs = get_directories(data_date, use_all_folders=True)
    print(f"Using all available date directories")
    
    # STEP 1: Train model with temporal filtering
    print("\nSTEP 1: TRAINING PREDICTION MODEL WITH TEMPORAL FILTERING")
    print("-" * 50)
    
    results = train_prediction_model(
        data_date=data_date, 
        save_outputs=save_outputs, 
        target_variable=target_variable,
        training_cutoff_days=training_cutoff_days
    )
    
    if not results or 'model' not in results:
        print("Error: Model training failed.")
        return None
    
    if save_outputs and 'prediction_path' in results:
        return results['prediction_path']
    return None

if __name__ == "__main__":
    
    TARGET_VARIABLE = 'pac'  
    TRAINING_CUTOFF_DAYS = 3  
    
    predictions_path = solar_prediction_pipeline(
        data_date=None,  
        save_outputs=True,
        target_variable=TARGET_VARIABLE,
        training_cutoff_days=TRAINING_CUTOFF_DAYS,
        visualization=False
    )
    
    if predictions_path:
        print(f"Pipeline terminé avec succès. Prédictions disponibles à: {predictions_path}")
    else:
        print("Le pipeline a rencontré des erreurs. Vérifiez les messages d'erreur ci-dessus.")
    






