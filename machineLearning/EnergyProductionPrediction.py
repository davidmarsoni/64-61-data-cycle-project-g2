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

# ---------------------------------
# Configuration
# ---------------------------------

def get_directories(data_date=None):
    """
    Get directory paths for data files.
    
    Parameters:
        data_date (str): Date string in 'YYYY-MM-DD' format, defaults to today if None
        
    Returns:
        dict: Dictionary of directory paths
    """
    if data_date is None:
        data_date = datetime.today().strftime('%Y-%m-%d')
    
    base_dir = os.getenv('BASE_DIR', 'C:/DataCollection')
    solar_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/Solarlogs')
    meteo_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/Meteo')
    conso_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/BellevueConso')
    
    return {
        'base_dir': base_dir,
        'solar_dir': solar_dir,
        'meteo_dir': meteo_dir,
        'conso_dir': conso_dir
    }

# ---------------------------------
# Data Loading and Preparation
# ---------------------------------

def load_csv_files(directory, file_pattern=None, date_filter=None):
    """
    Load multiple CSV files from a directory with optional pattern matching.
    
    Parameters:
        directory (str): Directory path containing CSV files
        file_pattern (str, optional): Pattern to match filenames (e.g., 'PV')
        date_filter (str, optional): Date to filter by (e.g., '2023-01-01')
        
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
            
        # Check if the file matches the date filter
        if date_filter and date_filter not in filename:
            continue
            
        # Load the file
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(file_path)
            
            # Add source filename as a column for reference
            df['source_file'] = filename
            
            all_data.append(df)
            print(f"Loaded {filename}, {len(df)} rows")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not all_data:
        print("No matching files found.")
        return pd.DataFrame()
    
    # Combine all dataframes
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


def prepare_prediction_data(pred_df):
    """
    Prepare weather prediction data.
    
    Parameters:
        pred_df (DataFrame): Raw prediction data
        
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
    
    # Set target_datetime as index
    df = df.set_index('target_datetime').sort_index()
    
    return df


def merge_datasets(pv_df, weather_df, pred_df=None, tolerance='30min'):
    """
    Merge PV, weather, and prediction data.
    
    Parameters:
        pv_df (DataFrame): Processed PV data
        weather_df (DataFrame): Processed weather data
        pred_df (DataFrame, optional): Processed prediction data
        tolerance (str): Time tolerance for merging asof
        
    Returns:
        DataFrame: Merged dataset
    """
    # Check if dataframes are empty
    if pv_df.empty or weather_df.empty:
        print("Warning: Cannot merge with empty dataframes.")
        return pd.DataFrame()
    
    # Reset indices for merging
    pv_reset = pv_df.reset_index()
    weather_reset = weather_df.reset_index()
    
    # Merge PV and weather data
    merged = pd.merge_asof(
        pv_reset.sort_values('datetime'),
        weather_reset.sort_values('datetime'),
        on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta(tolerance)
    )
    
    # If prediction data is provided, merge it as well
    if pred_df is not None and not pred_df.empty:
        # Add weather prediction columns to the merged dataframe
        final_df = merged.copy()
        
        # Identify prediction columns
        pred_columns = ['pred_glob_ctrl', 'pred_t_2m_ctrl', 'pred_relhum_2m_ctrl', 'pred_tot_prec_ctrl']
        pred_columns = [col for col in pred_columns if col in pred_df.columns]
        
        # Initialize columns with NaN
        for col in pred_columns:
            final_df[col] = np.nan
        
        final_df['prediction_horizon'] = np.nan
        
        # Convert prediction dataframe to a format easier to work with
        pred_data = pred_df.reset_index()
        
        # Make sure prediction data has appropriate datetime columns
        if 'target_datetime' in pred_data.columns:
            pred_data['date'] = pred_data['target_datetime'].dt.date
            pred_data['hour'] = pred_data['target_datetime'].dt.hour
        elif 'datetime' in pred_data.columns:
            pred_data['date'] = pred_data['datetime'].dt.date
            pred_data['hour'] = pred_data['datetime'].dt.hour
        else:
            print("Warning: Prediction data has no datetime column, skipping prediction merge")
            final_df = merged
        
        # If we have date and hour columns, we can proceed with merging
        if 'date' in pred_data.columns and 'hour' in pred_data.columns:
            # Add date and hour columns to merged data for easier matching
            final_df['date'] = final_df['datetime'].dt.date
            final_df['hour'] = final_df['datetime'].dt.hour
            
            # For each unique date and hour, find the best prediction
            for (date, hour), group in final_df.groupby(['date', 'hour']):
                # Find relevant predictions for this date and hour
                relevant_preds = pred_data[
                    (pred_data['date'] == date) & 
                    (pred_data['hour'] == hour)
                ]
                
                if not relevant_preds.empty and 'prediction' in relevant_preds.columns:
                    # Sort by prediction horizon (higher is better)
                    relevant_preds = relevant_preds.sort_values('prediction', ascending=False)
                    best_pred = relevant_preds.iloc[0]
                    
                    # Get indices of rows to update
                    indices = group.index
                    
                    # Update prediction values
                    for col in pred_columns:
                        if col in best_pred:
                            final_df.loc[indices, col] = best_pred[col]
                    
                    if 'prediction' in best_pred:
                        final_df.loc[indices, 'prediction_horizon'] = best_pred['prediction']
    else:
        final_df = merged
    
    # Add time-based features
    final_df['hour_sin'] = np.sin(2 * np.pi * final_df['datetime'].dt.hour / 24)
    final_df['hour_cos'] = np.cos(2 * np.pi * final_df['datetime'].dt.hour / 24)
    final_df['day_of_year'] = final_df['datetime'].dt.dayofyear / 365.0
    
    # Filter for daylight hours (simplistic approach - adjust as needed)
    if 'hour' not in final_df.columns:
        final_df['hour'] = final_df['datetime'].dt.hour
        
    daylight_df = final_df[(final_df['hour'] >= 6) & (final_df['hour'] <= 19)]
    
    # Rename the target column for clarity
    if 'value' in daylight_df.columns:
        daylight_df = daylight_df.rename(columns={'value': 'pv_output'})
    
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
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(model_info, f)
            print(f"Model saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None   


# ---------------------------------
# Main Execution Code
# ---------------------------------

def main_compare_models(data_date=None, save_outputs=True, target_variable='pac'):
    """
    Main function to run and compare two models:
    1. Complete model using all available features
    2. Prediction-only model using only weather predictions
    
    Parameters:
        data_date (str, optional): Date string in 'YYYY-MM-DD' format for data loading
        save_outputs (bool): Whether to save models and predictions to files
        target_variable (str): Target to predict - 'pac' for power or 'daysum' for daily energy
        
    Returns:
        dict: Dictionary containing both models' information and results
    """
    print("Solar Energy Production Prediction Model - Model Comparison")
    print("=" * 60)
    print(f"Target variable: {target_variable}")
    
    # Get directories
    dirs = get_directories(data_date)
    print(f"Using data directories:")
    for key, path in dirs.items():
        print(f"  {key}: {path}")
    
    # Load inverter data (min files)
    print("\nLoading inverter data...")
    inverter_files = load_csv_files(dirs['solar_dir'], file_pattern='min')
    
    if inverter_files.empty:
        print("No min files found. Trying PV files as fallback...")
        pv_files = load_csv_files(dirs['solar_dir'], file_pattern='-PV')
        if pv_files.empty:
            print("ERROR: No inverter or PV data found. Check file patterns and directory paths.")
            return None
        pv_data = prepare_pv_data(pv_files, target_variable)
    else:
        pv_data = prepare_pv_data(inverter_files, target_variable)
    
    if not pv_data.empty:
        print(f"PV data prepared: {pv_data.shape[0]} records")
    else:
        print("ERROR: Failed to prepare PV data. Check file formats and target variable.")
        return None
    
    # Load temperature data
    print("\nLoading temperature data...")
    temp_files = load_csv_files(dirs['conso_dir'], file_pattern='Temperature')
    
    # Load humidity data (if available)
    print("\nLoading humidity data...")
    humidity_files = load_csv_files(dirs['conso_dir'], file_pattern='Humidity')
    
    # Prepare weather data
    weather_data = prepare_weather_data(temp_files, humidity_files)
    if not weather_data.empty:
        print(f"Weather data loaded: {weather_data.shape[0]} records")
    
    # Load prediction data
    print("\nLoading prediction data...")
    pred_files = load_csv_files(dirs['meteo_dir'], file_pattern='Pred')
    pred_data = prepare_prediction_data(pred_files)
    if not pred_data.empty:
        print(f"Prediction data loaded: {pred_data.shape[0]} records")
    
    # For DaySum predictions, adjust data merging strategy
    if target_variable.lower() == 'daysum':
        # Need to adjust merge strategy for daily data
        print("Adapting merge strategy for daily predictions...")
        
        # Reset indices for merging
        pv_reset = pv_data.reset_index()
        
        # If we have only daily data, we need to get daily averages for weather
        if 'weather_data' in locals() and not weather_data.empty:
            weather_reset = weather_data.reset_index()
            weather_reset['date'] = weather_reset['datetime'].dt.date
            weather_daily = weather_reset.groupby('date').agg({
                'temperature': 'mean',
                'humidity': 'mean' if 'humidity' in weather_reset.columns else None
            }).dropna(axis=1).reset_index()
            weather_daily['datetime'] = pd.to_datetime(weather_daily['date'])
        
        # Handle prediction data similarly
        if 'pred_data' in locals() and not pred_data.empty:
            pred_reset = pred_data.reset_index()
            pred_reset['date'] = pred_reset['target_datetime'].dt.date
            
            # For each prediction variable, get daily averages during daylight hours (6-19)
            daylight_mask = (pred_reset['hour'] >= 6) & (pred_reset['hour'] <= 19)
            pred_daily = pred_reset[daylight_mask].groupby('date').agg({
                'pred_glob_ctrl': 'mean',
                'pred_t_2m_ctrl': 'mean',
                'pred_relhum_2m_ctrl': 'mean' if 'pred_relhum_2m_ctrl' in pred_reset.columns else None,
                'pred_tot_prec_ctrl': 'sum' if 'pred_tot_prec_ctrl' in pred_reset.columns else None
            }).dropna(axis=1).reset_index()
            pred_daily['datetime'] = pd.to_datetime(pred_daily['date'])
        
        # Merge datasets based on date instead of datetime
        merged_data = pv_reset
        
        if 'weather_daily' in locals():
            merged_data = pd.merge(
                merged_data,
                weather_daily,
                left_on='datetime',
                right_on='datetime',
                how='left'
            )
            
        if 'pred_daily' in locals():
            merged_data = pd.merge(
                merged_data,
                pred_daily,
                left_on='datetime',
                right_on='datetime',
                how='left'
            )
        
        # Add day of year feature
        merged_data['day_of_year'] = merged_data['datetime'].dt.dayofyear / 365.0
    else:
        # For PAC predictions, use the normal merge strategy
        print("\nMerging datasets...")
        merged_data = merge_datasets(pv_data, weather_data, pred_data)
    
    if not merged_data.empty:
        print(f"Merged data: {merged_data.shape[0]} records, {merged_data.shape[1]} columns")
        print(f"Columns: {merged_data.columns.tolist()}")
    else:
        print("ERROR: No data available after merging. Check data compatibility.")
        return None
    
    # Handle missing data
    print("\nHandling missing data...")
    clean_data = handle_missing_data(merged_data, method='interpolate')
    
    # Prepare features for full model
    print("\nPreparing full model features...")
    X_full, y = prepare_model_features(clean_data)
    
    # Prepare features for prediction-only model
    print("\nPreparing prediction-only model features...")
    X_pred_only, y_same = prepare_model_features_predictions_only(clean_data)
    
    results = {}
    
    # Train and evaluate full model
    if not X_full.empty and y is not None:
        print("\nTraining full model...")
        full_model_info = train_model(X_full, y, model_type='random_forest')
        full_predictions = predict_pv_output(full_model_info, clean_data)
        
        print("\nFull model metrics:")
        print(f"RMSE: {full_model_info['metrics']['RMSE']:.4f}")
        print(f"MAE: {full_model_info['metrics']['MAE']:.4f}")
        print(f"R2: {full_model_info['metrics']['R2']:.4f}")
        
        results['full_model'] = full_model_info
        results['full_predictions'] = full_predictions
        
        # Print full model feature importance
        if 'feature_importance' in full_model_info and full_model_info['feature_importance'] is not None:
            print("\nFull model feature importance:")
            print(full_model_info['feature_importance'])
    else:
        print("WARNING: Cannot train full model due to missing features or target.")
    
    # Train and evaluate prediction-only model
    if not X_pred_only.empty and y_same is not None:
        print("\nTraining prediction-only model...")
        pred_model_info = train_model(X_pred_only, y_same, model_type='random_forest')
        pred_predictions = predict_pv_output(pred_model_info, clean_data)
        
        print("\nPrediction-only model metrics:")
        print(f"RMSE: {pred_model_info['metrics']['RMSE']:.4f}")
        print(f"MAE: {pred_model_info['metrics']['MAE']:.4f}")
        print(f"R2: {pred_model_info['metrics']['R2']:.4f}")
        
        results['pred_model'] = pred_model_info
        results['pred_predictions'] = pred_predictions
        
        # Print prediction-only model feature importance
        if 'feature_importance' in pred_model_info and pred_model_info['feature_importance'] is not None:
            print("\nPrediction-only model feature importance:")
            print(pred_model_info['feature_importance'])
    else:
        print("WARNING: Cannot train prediction-only model due to missing features or target.")
    
    # Compare model performance
    if 'full_model' in results and 'pred_model' in results:
        print("\nModel Comparison:")
        print("-" * 40)
        print(f"Full model RMSE: {results['full_model']['metrics']['RMSE']:.4f}")
        print(f"Prediction-only model RMSE: {results['pred_model']['metrics']['RMSE']:.4f}")
        print(f"Performance difference: {results['pred_model']['metrics']['RMSE'] - results['full_model']['metrics']['RMSE']:.4f}")
        
        print(f"\nFull model R2: {results['full_model']['metrics']['R2']:.4f}")
        print(f"Prediction-only model R2: {results['pred_model']['metrics']['R2']:.4f}")
        print(f"Performance difference: {results['full_model']['metrics']['R2'] - results['pred_model']['metrics']['R2']:.4f}")
    
    # Visualize comparison
    if 'full_predictions' in results and 'pred_predictions' in results:
        # Scatter plots for actual vs predicted
        plt.figure(figsize=(14, 7))
        
        plt.subplot(1, 2, 1)
        plt.scatter(clean_data['pv_output'], results['full_predictions']['predicted_pv_output'], alpha=0.5)
        max_val = max(clean_data['pv_output'].max(), results['full_predictions']['predicted_pv_output'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Actual PV Output')
        plt.ylabel('Predicted PV Output')
        plt.title('Full Model Performance')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.scatter(clean_data['pv_output'], results['pred_predictions']['predicted_pv_output'], alpha=0.5)
        max_val = max(clean_data['pv_output'].max(), results['pred_predictions']['predicted_pv_output'].max())
        plt.plot([0, max_val], [0, max_val], 'r--')
        plt.xlabel('Actual PV Output')
        plt.ylabel('Predicted PV Output')
        plt.title('Prediction-Only Model Performance')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Time series comparison plot (for a sample period)
        if 'datetime' in clean_data.columns or (isinstance(clean_data.index, pd.DatetimeIndex)):
            # Get datetime column or use index
            if 'datetime' in clean_data.columns:
                datetime_col = clean_data['datetime']
            else:
                datetime_col = clean_data.index
                clean_data = clean_data.reset_index()
                
            # Get a representative sample period (e.g., 7 days)
            sample_start = datetime_col.min()
            sample_end = sample_start + pd.Timedelta(days=7)
            
            sample_mask = (datetime_col >= sample_start) & (datetime_col <= sample_end)
            sample_data = clean_data[sample_mask].copy()
            
            if not sample_data.empty:
                plt.figure(figsize=(15, 6))
                plt.plot(sample_data['datetime'], sample_data['pv_output'], 'k-', label='Actual')
                plt.plot(sample_data['datetime'], results['full_predictions'].loc[sample_data.index, 'predicted_pv_output'], 'b-', label='Full Model')
                plt.plot(sample_data['datetime'], results['pred_predictions'].loc[sample_data.index, 'predicted_pv_output'], 'r-', label='Pred-Only Model')
                
                plt.xlabel('Date/Time')
                plt.ylabel('PV Output')
                plt.title('Model Comparison - Time Series (Sample Period)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
# Calculate and display error metrics by hour of day
                if 'hour' in sample_data.columns:
                    sample_data['full_error'] = abs(sample_data['pv_output'] - results['full_predictions'].loc[sample_data.index, 'predicted_pv_output'])
                    sample_data['pred_error'] = abs(sample_data['pv_output'] - results['pred_predictions'].loc[sample_data.index, 'predicted_pv_output'])
                    
                    # Group by hour and calculate mean error
                    hourly_errors = sample_data.groupby('hour').agg({
                        'full_error': 'mean',
                        'pred_error': 'mean',
                        'pv_output': 'mean'  # For reference
                    }).reset_index()
                    
                    # Plot hourly errors
                    plt.figure(figsize=(12, 6))
                    plt.bar(hourly_errors['hour'] - 0.2, hourly_errors['full_error'], width=0.4, label='Full Model Error', alpha=0.7, color='blue')
                    plt.bar(hourly_errors['hour'] + 0.2, hourly_errors['pred_error'], width=0.4, label='Pred-Only Model Error', alpha=0.7, color='red')
                    plt.plot(hourly_errors['hour'], hourly_errors['pv_output'], 'k--', label='Avg PV Output')
                    
                    plt.xlabel('Hour of Day')
                    plt.ylabel('Mean Absolute Error')
                    plt.title('Error Comparison by Hour of Day')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.xticks(range(0, 24))
                    plt.tight_layout()
                    plt.show()
    
    # Test models on future prediction data only
    # This simulates how the model would perform in deployment
    if 'pred_data' in locals() and not pred_data.empty:
        print("\nTesting models on prediction data only (deployment simulation)...")
        
        # Prepare features using only prediction data
        future_data = pred_data.copy()
        
        # Add time-based features
        future_data['hour_sin'] = np.sin(2 * np.pi * future_data.index.hour / 24)
        future_data['hour_cos'] = np.cos(2 * np.pi * future_data.index.hour / 24)
        future_data['day_of_year'] = future_data.index.dayofyear / 365.0
        
        # Select a recent date range to simulate future predictions
        recent_mask = future_data.index >= (future_data.index.max() - pd.Timedelta(days=7))
        future_sample = future_data[recent_mask]
        
        if not future_sample.empty:
            # Make predictions using only the prediction-only model
            # (Skip full model since it requires measurements we won't have)
            if 'pred_model' in results:
                pred_future_predictions = predict_pv_output(results['pred_model'], future_sample)
                print("Generated prediction-only model predictions for simulated future data.")
                
                # Plot predictions for future data
                plt.figure(figsize=(15, 6))
                plt.plot(pred_future_predictions.index, pred_future_predictions['predicted_pv_output'], 'r-', 
                        label='Pred-Only Model Prediction')
                
                plt.xlabel('Date/Time')
                plt.ylabel('Predicted PV Output')
                plt.title('Future Predictions (Deployment Simulation)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
    
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
        
        # Save models if available
        if 'full_model' in results:
            full_model_path = os.path.join(model_dir, f"full_model_{target_suffix}_{current_date}.pkl")
            save_model(results['full_model'], output_path=full_model_path)
            print(f"Full model saved to: {full_model_path}")
        
        if 'pred_model' in results:
            pred_model_path = os.path.join(model_dir, f"pred_only_model_{target_suffix}_{current_date}.pkl")
            save_model(results['pred_model'], output_path=pred_model_path)
            print(f"Prediction-only model saved to: {pred_model_path}")
        
        # Save predictions if available
        if 'full_predictions' in results:
            full_pred_path = os.path.join(output_dir, f"full_model_predictions_{target_suffix}_{current_date}.csv")
            save_predictions_to_csv(results['full_predictions'], output_path=full_pred_path)
            print(f"Full model predictions saved to: {full_pred_path}")
        
        if 'pred_predictions' in results:
            pred_pred_path = os.path.join(output_dir, f"pred_only_predictions_{target_suffix}_{current_date}.csv")
            save_predictions_to_csv(results['pred_predictions'], output_path=pred_pred_path)
            print(f"Prediction-only model predictions saved to: {pred_pred_path}")
        
        
        if 'pred_future_predictions' in locals():
            pred_future_path = os.path.join(output_dir, f"pred_only_future_predictions_{target_suffix}_{current_date}.csv")
            save_predictions_to_csv(pred_future_predictions, output_path=pred_future_path)
            print(f"Prediction-only model future predictions saved to: {pred_future_path}")
    
    # Generate detailed report
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY REPORT")
    print("="*60)
    
    if 'full_model' in results and 'pred_model' in results:
        performance_diff = results['pred_model']['metrics']['RMSE'] - results['full_model']['metrics']['RMSE']
        relative_diff_pct = (performance_diff / results['full_model']['metrics']['RMSE']) * 100
        
        print(f"Model Type: {target_variable.upper()} Prediction")
        print(f"Full Model Features: {', '.join(results['full_model']['features'])}")
        print(f"Pred-Only Features: {', '.join(results['pred_model']['features'])}")
        print("\nPerformance Metrics:")
        print(f"  Full Model RMSE: {results['full_model']['metrics']['RMSE']:.4f}")
        print(f"  Pred-Only RMSE: {results['pred_model']['metrics']['RMSE']:.4f}")
        print(f"  Absolute Difference: {performance_diff:.4f}")
        print(f"  Relative Difference: {relative_diff_pct:.2f}%")
        print(f"\n  Full Model R²: {results['full_model']['metrics']['R2']:.4f}")
        print(f"  Pred-Only R²: {results['pred_model']['metrics']['R2']:.4f}")
        print(f"  Difference: {results['full_model']['metrics']['R2'] - results['pred_model']['metrics']['R2']:.4f}")
        
        # Recommendation
        if relative_diff_pct <= 10:
            print("\nRECOMMENDATION: The prediction-only model performs within 10% of the full model.")
            print("You can confidently deploy the prediction-only model for operational use.")
        elif relative_diff_pct <= 20:
            print("\nRECOMMENDATION: The prediction-only model shows moderate degradation compared to the full model.")
            print("Consider if this level of performance is acceptable for your application.")
        else:
            print("\nRECOMMENDATION: The prediction-only model shows significant degradation compared to the full model.")
            print("Consider enhancing the prediction-only model with additional derived features or using a more complex model architecture.")
    
    return results
    


if __name__ == "__main__":
    # Specify which target variable to predict: 'pac' or 'daysum'
    target = 'pac'  # Change to 'daysum' for daily energy prediction
    
    # You can specify a data date here
    results = main_compare_models(data_date='2025-03-14',save_outputs=True, target_variable=target)