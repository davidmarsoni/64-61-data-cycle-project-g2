import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---------------------------------
# Functions for data loading
# ---------------------------------

def get_directories(data_date=None):
    """
    Retrieves directory paths for data files.
    
    Parameters:
        data_date (str): Date in 'YYYY-MM-DD' format, uses current date by default
        
    Returns:
        dict: Dictionary of directory paths
    """
    if data_date is None:
        data_date = datetime.today().strftime('%Y-%m-%d')
    
    base_dir = os.getenv('BASE_DIR', 'C:/DataCollection')
    meteo_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/Meteo')
    conso_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/BellevueConso')
    room_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/BellevueBooking')
    
    return {
        'base_dir': base_dir,
        'meteo_dir': meteo_dir,
        'conso_dir': conso_dir,
        'room_dir': room_dir
    }

def load_csv_files(directory, file_pattern=None):
    """
    Loads multiple CSV files from a directory with optional pattern matching.
    
    Parameters:
        directory (str): Path to directory containing CSV files
        file_pattern (str, optional): Pattern to filter file names
        
    Returns:
        DataFrame: Combined data from all matching CSV files
    """
    all_data = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist.")
        return pd.DataFrame()
    
    # List all CSV files in the directory
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        # Check if the file matches the pattern
        if file_pattern and file_pattern not in filename:
            continue
            
        # Load the file with multiple encoding attempts
        file_path = os.path.join(directory, filename)
        df = None
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                # For room allocations, set low_memory=False
                if 'RoomAllocations' in filename:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                else:
                    df = pd.read_csv(file_path, encoding=encoding)
                
                # Add the source file name as a column for reference
                df['source_file'] = filename
                
                all_data.append(df)
                print(f"Loaded {filename} with encoding {encoding}, {len(df)} rows")
                break  # Successful load, exit encoding loop
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                break
    
    if not all_data:
        print("No matching files found.")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    return combined_df

def load_consumption_data(consumption_df):
    """
    Prepares energy consumption data.
    
    Parameters:
        consumption_df (DataFrame): Raw consumption data
        
    Returns:
        DataFrame: Processed consumption data
    """
    # Create a copy to avoid modifying the original
    df = consumption_df.copy()
    
    # Check if the dataframe is empty
    if df.empty:
        print("Warning: Consumption dataframe is empty.")
        return df
    
    # Standardize column names (lowercase)
    df.columns = [col.lower() for col in df.columns]
    
    # Check required columns
    required_cols = ['date', 'time', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing required columns in consumption data: {missing_cols}")
        return pd.DataFrame()
    
    # Create the datetime column
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
    
    # Drop rows with invalid datetime
    invalid_dates = df['datetime'].isna()
    if invalid_dates.any():
        print(f"Warning: Dropping {invalid_dates.sum()} rows with invalid datetime")
        df = df[~invalid_dates].copy()
    
    # Rename 'value' to 'consumption' for clarity
    df = df.rename(columns={'value': 'consumption'})
    
    # Set datetime as index
    df = df.set_index('datetime').sort_index()
    
    return df[['consumption']]

def load_weather_data(temp_df, humidity_df=None):
    """
    Prepares and combines temperature and humidity data.
    
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
            temp_df['datetime'] = pd.to_datetime(temp_df['date'] + ' ' + temp_df['time'], errors='coerce')
            temp_df = temp_df.rename(columns={'value': 'temperature'})
            temp_df = temp_df.drop_duplicates(subset=['datetime'])
            temp_df = temp_df.set_index('datetime').sort_index()
        else:
            print("Warning: Temperature data with missing columns.")
            return pd.DataFrame()
    else:
        print("Warning: Temperature dataframe is empty.")
        return pd.DataFrame()
    
    # If humidity data is provided, process and merge it
    if humidity_df is not None and not humidity_df.empty:
        humidity_df = humidity_df.copy()
        humidity_df.columns = [col.lower() for col in humidity_df.columns]
        
        if all(col in humidity_df.columns for col in ['date', 'time', 'value']):
            humidity_df['datetime'] = pd.to_datetime(humidity_df['date'] + ' ' + humidity_df['time'], errors='coerce')
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
            print("Warning: Humidity data with missing columns.")
            weather_df = temp_df[['temperature']]
    else:
        # Use only temperature data
        weather_df = temp_df[['temperature']]
    
    # Add enhanced weather features
    if 'temperature' in weather_df.columns:
        # Heating and cooling degree days
        weather_df['heating_degree'] = weather_df['temperature'].apply(lambda x: max(18.0 - x, 0))  # Base 18°C for heating
        weather_df['cooling_degree'] = weather_df['temperature'].apply(lambda x: max(x - 22.0, 0))  # Base 22°C for cooling
    
    return weather_df

def load_prediction_weather_data(pred_files, select_highest_prediction=True):
    """
    Prepares weather prediction data and selects the highest prediction (most recent).
    
    Parameters:
        pred_files (DataFrame): Raw weather prediction data
        select_highest_prediction (bool): If True, selects prediction with highest value
        
    Returns:
        DataFrame: Processed weather data
    """
    if pred_files.empty:
        print("Warning: Weather prediction dataframe is empty.")
        return pd.DataFrame()
    
    try:
        # Copy the data
        df = pred_files.copy()
        
        # Standardize column names (lowercase)
        original_columns = df.columns.tolist()
        print(f"Original prediction columns: {original_columns}")
        
        # Convert column names to standardized versions
        df.columns = [col.lower() for col in df.columns]
        
        # Rename columns for consistency
        column_mapping = {
            'pred_t_2m_ctrl': 'temperature',
            'pred_relhum_2m_ctrl': 'humidity',
            'pred_tot_prec_ctrl': 'precipitation',
            'pred_glob_ctrl': 'global_radiation'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Create datetime from date and time
        if 'date' in df.columns and 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], errors='coerce')
            
            # Check if we have the 'prediction' column
            if 'prediction' in df.columns and select_highest_prediction:
                print("Selecting most recent predictions (highest 'prediction' value)...")
                
                # Number of predictions before filtering
                total_pred_count = len(df)
                
                # Group by datetime and select the prediction with the HIGHEST value
                df_grouped = df.groupby(['datetime']).apply(
                    lambda x: x.loc[x['prediction'].idxmax()]
                )
                
                # Check if the result is empty
                if df_grouped.empty:
                    print("Warning: No valid predictions found after grouping")
                    return pd.DataFrame()
                
                # Convert to DataFrame and reset index
                if isinstance(df_grouped.index, pd.MultiIndex):
                    df = df_grouped.reset_index(level=0, drop=True)
                else:
                    df = pd.DataFrame(df_grouped)
                
                # Number of predictions after filtering
                filtered_pred_count = len(df)
                print(f"Number of predictions reduced from {total_pred_count} to {filtered_pred_count} (selecting most recent for each hour)")
            
            # Set datetime as index
            df = df.set_index('datetime').sort_index()
            
            # Select relevant columns
            weather_cols = ['temperature', 'humidity', 'precipitation', 'global_radiation']
            available_cols = [col for col in weather_cols if col in df.columns]
            
            if not available_cols:
                print("Warning: No relevant weather columns found.")
                return pd.DataFrame()
            
            weather_df = df[available_cols]
            
            # Add enhanced weather features
            if 'temperature' in weather_df.columns:
                # Heating and cooling degree days
                weather_df['heating_degree'] = weather_df['temperature'].apply(lambda x: max(18.0 - x, 0))
                weather_df['cooling_degree'] = weather_df['temperature'].apply(lambda x: max(x - 22.0, 0))
            
            return weather_df
            
        else:
            print("Warning: Missing 'date' or 'time' columns in prediction data")
            return pd.DataFrame()
    
    except Exception as e:
        print(f"Error processing weather prediction data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
    
    except Exception as e:
        print(f"Error processing weather prediction data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def load_latest_room_allocation_data(directory):
    """
    Loads only the most recent room allocation file.
    
    Parameters:
        directory (str): Path to directory containing booking files
        
    Returns:
        DataFrame: Room occupancy data
    """
    try:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist.")
            return pd.DataFrame()
        
        # List all CSV files in the directory that match the pattern
        room_files = [f for f in os.listdir(directory) if f.endswith('.csv') and 'RoomAllocations' in f]
        
        if not room_files:
            print("No room allocation files found.")
            return pd.DataFrame()
        
        # Sort files by date (assuming format RoomAllocations_YYYYMMDD.csv)
        room_files.sort(reverse=True)  # Sort descending to get the most recent first
        
        # Load only the most recent file
        latest_file = room_files[0]
        file_path = os.path.join(directory, latest_file)
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                print(f"Loaded {latest_file} with encoding {encoding}, {len(df)} rows")
                break  # Successful load
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error loading {latest_file}: {e}")
                return pd.DataFrame()
        
        # If no encoding worked
        if df is None:
            print(f"Unable to load {latest_file} with available encodings.")
            return pd.DataFrame()
        
        # Prepare the data as in the original function
        df.columns = [col.lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        # Group by date and count bookings
        daily_occupancy = df.groupby('date').size().reset_index()
        daily_occupancy.columns = ['date', 'rooms_occupied']
        
        # Convert to hourly format
        daily_occupancy['datetime'] = pd.to_datetime(daily_occupancy['date'])
        daily_occupancy = daily_occupancy.set_index('datetime')
        daily_occupancy = daily_occupancy.drop(columns=['date'])
        
        # Convert to hourly format (each hour of the day will have the same value)
        min_date = daily_occupancy.index.min()
        max_date = daily_occupancy.index.max()
        
        hourly_index = pd.date_range(start=min_date, end=max_date + pd.Timedelta(days=1), freq='h')[:-1]
        hourly_df = pd.DataFrame(index=hourly_index)
        hourly_df['day'] = hourly_df.index.date
        
        # Merge with daily data
        daily_occupancy['day'] = daily_occupancy.index.date
        hourly_df = pd.merge(
            hourly_df,
            daily_occupancy[['day', 'rooms_occupied']],
            on='day',
            how='left'
        )
        
        # Clean up and set index
        hourly_df = hourly_df.drop(columns=['day'])
        hourly_df.index.name = 'datetime'
        
        return hourly_df
        
    except Exception as e:
        print(f"Error processing room allocation data: {e}")
        return pd.DataFrame()

def load_data_from_directories(data_date=None):
    """
    Loads all relevant data from standard directories.
    """
    # Get directories
    dirs = get_directories(data_date)
    print(f"Using data directories:")
    for key, path in dirs.items():
        print(f"  {key}: {path}")
    
    # 1. Load consumption data
    print("\nLoading consumption data...")
    consumption_files = load_csv_files(dirs['conso_dir'], file_pattern='Consumption')
    
    consumption_df = pd.DataFrame()
    if not consumption_files.empty:
        # Prepare consumption data
        consumption_df = load_consumption_data(consumption_files)
        print(f"Prepared consumption data: {len(consumption_df)} records")
    
    # 2. Load weather data
    print("\nLoading weather data...")
    
    # Try both possible formats for weather data
    temp_files = load_csv_files(dirs['conso_dir'], file_pattern='Temperature')
    humidity_files = load_csv_files(dirs['conso_dir'], file_pattern='Humidity')
    
    # Also try to load weather prediction files
    pred_files = load_csv_files(dirs['meteo_dir'], file_pattern='Pred_')
    
    weather_df = pd.DataFrame()
    if not temp_files.empty and not humidity_files.empty:
        # Use temperature and humidity files if available
        weather_df = load_weather_data(temp_files, humidity_files)
        print(f"Loaded weather data (temp + humidity): {len(weather_df)} records")
    elif not pred_files.empty:
        # Otherwise, use weather prediction files
        weather_df = load_prediction_weather_data(pred_files)
        print(f"Loaded weather data (prediction): {len(weather_df)} records")
    else:
        print("No weather data available.")
    
    # 3. Load room allocation data
    print("\nLoading room allocation data (most recent file)...")
    room_df = load_latest_room_allocation_data(dirs['room_dir'])

    if not room_df.empty:
        print(f"Prepared room allocation data: {len(room_df)} records")
    else:
        print("No room allocation data available.")
    
    # Return all DataFrames
    return {
        'consumption': consumption_df,
        'weather': weather_df,
        'room': room_df
    }

# ---------------------------------
# Functions for data preparation for window sliding
# ---------------------------------

def merge_datasets_smart(consumption_df, weather_df, room_df=None):
    """
    Merges datasets intelligently with handling of missing values.
    
    Parameters:
        consumption_df (DataFrame): Consumption data
        weather_df (DataFrame): Weather data
        room_df (DataFrame, optional): Room occupancy data
        
    Returns:
        DataFrame: Merged data
    """
    print("Intelligent data merging...")
    
    # Check that the dataframes are not empty
    if consumption_df.empty:
        print("Error: Missing consumption data.")
        return pd.DataFrame()
    
    # Create a copy of the consumption data
    result = consumption_df.copy()
    
    # 1. Merge with weather data
    if not weather_df.empty:
        # Identify numeric columns
        weather_numeric = weather_df.select_dtypes(include=[np.number]).copy()
        
        # Merge
        print(f"Merging consumption ({result.shape}) with weather ({weather_numeric.shape})...")
        result = pd.merge(
            result,
            weather_numeric,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Interpolate missing values for weather
        for col in weather_numeric.columns:
            if result[col].isna().any():
                # Use time interpolation (preserves trends)
                result[col] = result[col].interpolate(method='time').bfill().ffill()
        
        print(f"After weather merge: {result.shape}")
    
    # 2. Merge with room occupancy data
    if room_df is not None and not room_df.empty:
        # Simplification: keep only the occupancy column
        room_simple = room_df[['rooms_occupied']].copy()
        
        print(f"Merging with occupancy data ({room_simple.shape})...")
        result = pd.merge(
            result,
            room_simple,
            left_index=True,
            right_index=True,
            how='left'
        )
        
        # Fill missing values for occupancy
        if 'rooms_occupied' in result.columns and result['rooms_occupied'].isna().any():
            # Use 0 as default value (no occupancy)
            result['rooms_occupied'] = result['rooms_occupied'].fillna(0)
        
        print(f"After occupancy merge: {result.shape}")
    
    # Check remaining missing values
    missing_values = result.isna().sum()
    if missing_values.sum() > 0:
        print("Missing values after merge:")
        print(missing_values[missing_values > 0])
        
        # Fill remaining missing values
        result = result.interpolate(method='time').bfill().ffill()
    
    return result

def add_time_features(df):
    """
    Adds basic time features.
    """
    df = df.copy()
    
    # Check if the index is a DatetimeIndex, otherwise convert it
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Warning: Index is not a DatetimeIndex. Attempting conversion...")
        try:
            # If the index contains dates as strings, try to convert it
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            print(f"Error converting index to DatetimeIndex: {e}")
            print("Creating a new time index...")
            # If conversion fails, create a new index
            old_index = df.index
            df = df.reset_index(drop=True)
            # Save the old index as a column if necessary
            if not pd.api.types.is_datetime64_any_dtype(old_index):
                df['original_index'] = old_index
    
    # Now that we have a DatetimeIndex, extract the components
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    
    # Binary indicators
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_working_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & 
                             (df['is_weekend'] == 0)).astype(int)
    
    # Cyclical representation of time variables
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_small_window_features(df, target_col='consumption', window_sizes=[1, 3, 6, 12, 24]):
    """
    Creates sliding window features with smaller sizes
    and intelligent handling of missing values.
    """
    print(f"Creating sliding window features with reduced windows for {len(df)} rows...")
    df_windows = df.copy()
    
    # For each window size, create lag features
    for window in window_sizes:
        print(f"Processing window size {window}...")
        
        # Lag value
        df_windows[f'{target_col}_lag_{window}'] = df_windows[target_col].shift(window)
        
        # Moving average (trend indicator)
        df_windows[f'{target_col}_rolling_{window}_mean'] = df_windows[target_col].shift(1).rolling(window=window, min_periods=1).mean()
    
    # Create a subset of essential features (those with fewer null values)
    essential_cols = [target_col]
    for window in [1, 3]:  # Smallest windows
        essential_cols.extend([
            f'{target_col}_lag_{window}', 
            f'{target_col}_rolling_{window}_mean'
        ])
    
    # Diagnose missing values
    null_counts = df_windows.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    
    if not cols_with_nulls.empty:
        print(f"Columns with missing values:")
        for col, count in cols_with_nulls.items():
            print(f"  {col}: {count} missing values")
    
    # Missing value handling strategy:
    # 1. First, handle non-essential columns
    for col in df_windows.columns:
        if col not in essential_cols and df_windows[col].isna().any():
            if pd.api.types.is_numeric_dtype(df_windows[col]):
                # Interpolate then fill with mean
                temp_fill = df_windows[col].interpolate(method='linear')
                temp_fill = temp_fill.fillna(temp_fill.mean())
                df_windows[col] = temp_fill
    
    # 2. Then, only drop rows where essential columns have null values
    print(f"Dropping rows with null values in {len(essential_cols)} essential columns...")
    n_before = len(df_windows)
    df_windows = df_windows.dropna(subset=essential_cols)
    n_after = len(df_windows)
    
    print(f"Number of rows dropped: {n_before - n_after}")
    print(f"Result: {len(df_windows)} usable rows after creating window features")
    
    return df_windows

# ---------------------------------
# Modeling functions
# ---------------------------------

def train_test_split_timeseries(df, test_size=0.2):
    """
    Splits data into training and testing sets chronologically.
    """
    # Check if the DataFrame is empty
    if df.empty:
        print("Warning: Empty DataFrame for chronological split")
        empty_df = pd.DataFrame(columns=df.columns)
        return empty_df, empty_df
    
    # Sort by time index
    df_sorted = df.sort_index()
    
    # Calculate split indices
    n = len(df_sorted)
    test_idx = int(n * (1 - test_size))
    
    # Split the data
    train_df = df_sorted.iloc[:test_idx].copy()
    test_df = df_sorted.iloc[test_idx:].copy()
    
    print(f"Chronological split: train={len(train_df)}, test={len(test_df)}")
    
    return train_df, test_df

def train_model(X_train, y_train):
    """
    Trains a simple and robust RandomForest model.
    """
    if X_train.empty or len(y_train) == 0:
        print("ERROR: Empty training data.")
        return None
    
    print(f"Training RandomForest model with {X_train.shape[1]} features on {len(X_train)} examples...")
    
    # Use RandomForest which is more robust and does not require scaling
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return {
        'model': model,
        'feature_importance': feature_importance
    }

def plot_predictions(y_test, y_pred, title="Consumption Prediction"):
    """
    Visualizes predictions compared to actual values.
    """
    if len(y_test) == 0 or len(y_pred) == 0:
        print("Error: No data for visualization.")
        return
    
    # For time series
    if isinstance(y_test.index, pd.DatetimeIndex):
        # Create a dataframe to facilitate plotting
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Prediction': y_pred
        }, index=y_test.index)
        
        # 1. Plot an overall view of the predictions
        plt.figure(figsize=(15, 7))
        plt.plot(results_df.index, results_df['Actual'], label='Actual Consumption', color='blue', alpha=0.7)
        plt.plot(results_df.index, results_df['Prediction'], label='Prediction', color='red', alpha=0.7)
        plt.title(f"{title} - Overall View")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 2. Plot a zoomed-in view of one week of data (if available)
        if len(results_df) > 24*7:
            # Select the last week
            last_week = results_df.iloc[-24*7:]
            
            plt.figure(figsize=(15, 7))
            plt.plot(last_week.index, last_week['Actual'], label='Actual Consumption', color='blue', alpha=0.7)
            plt.plot(last_week.index, last_week['Prediction'], label='Prediction', color='red', alpha=0.7)
            plt.title(f"{title} - Last Week")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # 3. Aggregate by day
        daily_df = results_df.groupby(results_df.index.date).sum()
        
        plt.figure(figsize=(15, 7))
        dates = [pd.Timestamp(date) for date in daily_df.index]
        plt.plot(dates, daily_df['Actual'], label='Actual Consumption', marker='o', color='blue')
        plt.plot(dates, daily_df['Prediction'], label='Prediction', marker='x', color='red')
        plt.title(f"{title} - Daily Aggregation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Scatter plot (actual vs predicted values)
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Consumption')
    plt.ylabel('Predicted Consumption')
    plt.title(f"{title} - Actual vs Predicted")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def plot_feature_importance(model_info, top_n=20):
    """
    Visualizes feature importance for the model.
    """
    if model_info is None or 'feature_importance' not in model_info:
        print("Error: Feature importance information not available.")
        return
    
    importance_df = model_info['feature_importance'].head(top_n)
    
    plt.figure(figsize=(12, 8))
    plt.barh(
        importance_df['Feature'][::-1],
        importance_df['Importance'][::-1]
    )
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Relative Importance')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

def evaluate_model(model_info, X_test, y_test):
    """
    Evaluates the model on the test set.
    """
    if model_info is None or X_test.empty or len(y_test) == 0:
        print("Warning: Unable to evaluate model (missing data)")
        return None
    
    model = model_info['model']
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nEvaluation metrics on {len(y_test)} test examples:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # Calculate daily metrics
    if isinstance(y_test.index, pd.DatetimeIndex):
        # Create DataFrames to group
        test_df = pd.DataFrame({'actual': y_test, 'predicted': y_pred}, index=y_test.index)
        
        # Group by day
        daily_actual = test_df.groupby(test_df.index.date)['actual'].sum()
        daily_pred = test_df.groupby(test_df.index.date)['predicted'].sum()
        
        # Daily metrics
        daily_rmse = np.sqrt(mean_squared_error(daily_actual, daily_pred))
        daily_mae = mean_absolute_error(daily_actual, daily_pred)
        daily_r2 = r2_score(daily_actual, daily_pred)
        
        print(f"\nDaily metrics:")
        print(f"  Daily RMSE: {daily_rmse:.4f}")
        print(f"  Daily MAE: {daily_mae:.4f}")
        print(f"  Daily R²: {daily_r2:.4f}")
    
    # Prepare evaluation results
    eval_results = {
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'predictions': y_pred,
        'actual': y_test
    }
    
    # Add daily metrics if calculated
    if 'daily_rmse' in locals():
        eval_results['metrics']['daily_rmse'] = daily_rmse
        eval_results['metrics']['daily_mae'] = daily_mae
        eval_results['metrics']['daily_r2'] = daily_r2
    
    return eval_results

def predict_aggregate_daily(data_date='2025-03-14', window_sizes=[1, 3, 6, 12, 24], save_outputs=True):
    """
    Hybrid function: hourly prediction then daily aggregation.
    """
    print(f"Hybrid model: hourly prediction then daily aggregation")
    print(f"Windows used (in hours): {window_sizes}")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading data...")
    data = load_data_from_directories(data_date)
    
    if data['consumption'].empty:
        print("ERROR: No consumption data found.")
        return None
    
    print(f"Data loaded: {len(data['consumption'])} consumption records")
    
    # 2. Merge datasets (as in your original hourly model)
    print("\n2. Merging datasets...")
    merged_data = merge_datasets_smart(data['consumption'], data['weather'], data['room'])
    
    if merged_data.empty:
        print("ERROR: Data merge failed.")
        return None
    
    print(f"Merged data: {len(merged_data)} records, {merged_data.shape[1]} columns")
    
    # 3. Add time features
    print("\n3. Adding time features...")
    # Use the corrected add_time_features function
    df_with_time = add_time_features(merged_data)
    
    # 4. Add sliding window features
    print("\n4. Creating hourly sliding window features...")
    df_windows = create_small_window_features(df_with_time, window_sizes=window_sizes)
    
    if df_windows.empty:
        print("ERROR: All data lost during feature creation.")
        return None
    
    print(f"Final dataset: {df_windows.shape[0]} rows, {df_windows.shape[1]} columns")
    
    # 5. Prepare features and target for the model
    X = df_windows.drop(columns=['consumption'])
    y = df_windows['consumption']
    
    # 6. Chronological data split
    print("\n5. Chronological data split...")
    X_train, X_test = train_test_split_timeseries(X)
    y_train, y_test = train_test_split_timeseries(y)
    
    # 7. Train the hourly model
    print("\n6. Training the hourly model...")
    model_info = train_model(X_train, y_train)
    
    if model_info is None:
        print("ERROR: Model training failed.")
        return None
    
    # 8. Evaluate the hourly model
    print("\n7. Evaluating the hourly model...")
    hourly_eval = evaluate_model(model_info, X_test, y_test)
    
    if hourly_eval is None:
        print("ERROR: Model evaluation failed.")
        return None
    
    # 9. Aggregate predictions to daily scale
    print("\n8. Aggregating predictions to daily scale...")
    
    # Create a dataframe with hourly predictions
    hourly_results = pd.DataFrame({
        'actual': y_test,
        'predicted': hourly_eval['predictions']
    })
    
    # Add the date (day only) for aggregation
    hourly_results['date'] = hourly_results.index.date
    
    # Aggregate actual and predicted values by day
    daily_results = hourly_results.groupby('date').agg({
        'actual': 'sum',
        'predicted': 'sum'
    })
    
    # Calculate evaluation metrics for daily predictions
    daily_rmse = np.sqrt(mean_squared_error(daily_results['actual'], daily_results['predicted']))
    daily_mae = mean_absolute_error(daily_results['actual'], daily_results['predicted'])
    daily_r2 = r2_score(daily_results['actual'], daily_results['predicted'])
    daily_mape = np.mean(np.abs((daily_results['actual'] - daily_results['predicted']) / daily_results['actual'])) * 100
    
    print(f"\nEvaluation metrics on {len(daily_results)} test days:")
    print(f"  Daily RMSE: {daily_rmse:.2f}")
    print(f"  Daily MAE: {daily_mae:.2f}")
    print(f"  Daily R²: {daily_r2:.4f}")
    print(f"  Daily MAPE: {daily_mape:.2f}%")
    
    # 10. Visualizations
    print("\n9. Creating visualizations...")
    
    # Visualize feature importance
    plot_feature_importance(model_info, top_n=15)
    
    # Visualize daily predictions
    plt.figure(figsize=(15, 7))
    plt.plot(daily_results.index, daily_results['actual'], 'o-', label='Actual Consumption', color='blue', alpha=0.7)
    plt.plot(daily_results.index, daily_results['predicted'], 'x-', label='Prediction', color='red', alpha=0.7)
    plt.title("Daily Consumption Prediction (Hybrid Model)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 11. Save results if requested
    if save_outputs:
        print("\n10. Saving model and predictions...")
        
        # Save predictions to CSV
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Save hourly predictions
        hourly_csv_path = os.path.join(output_dir, f"hourly_predictions_{current_date}.csv")
        hourly_results.to_csv(hourly_csv_path)
        
        # Save daily predictions
        daily_csv_path = os.path.join(output_dir, f"daily_aggregated_predictions_{current_date}.csv")
        daily_results.to_csv(daily_csv_path)
        
        # Save the model
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"hybrid_model_{current_date}.pkl")
        
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Hourly predictions saved to: {hourly_csv_path}")
        print(f"Daily predictions saved to: {daily_csv_path}")
        print(f"Model saved to: {model_path}")
    
    # 12. Return results
    return {
        'model_info': model_info,
        'hourly_evaluation': hourly_eval,
        'daily_evaluation': {
            'metrics': {
                'rmse': daily_rmse,
                'mae': daily_mae,
                'r2': daily_r2,
                'mape': daily_mape
            },
            'predictions': daily_results
        },
        'data': {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    }

# Execute the main function
if __name__ == "__main__":
    try:
        print("Starting hybrid prediction model...")
        
        # Use specified date and windows adapted to hourly data
        results = predict_aggregate_daily(
            data_date='2025-03-14',
            window_sizes=[1, 3, 6, 12, 24],  # Windows adapted to hourly data
            save_outputs=True
        )
        
        print("\nExecution completed successfully.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()