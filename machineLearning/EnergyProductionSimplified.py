# Solar Energy Production Prediction Model - Amélioré
# ------------------------------------------------------------------------
# This model predicts solar panel energy production based on weather forecasts.
# Simplified version with improved performance, using only MIN and PRED files.

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---------------------------------
# Configuration
# ---------------------------------

def get_directories(data_date=None):
    """Get directory paths for data files."""
    if data_date is None:
        data_date = datetime.today().strftime('%Y-%m-%d')
    
    base_dir = os.getenv('BASE_DIR', 'C:/DataCollection')
    solar_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/Solarlogs')
    meteo_dir = os.path.join(base_dir, f'cleaned_data_{data_date}/Meteo')
    
    return {
        'base_dir': base_dir,
        'solar_dir': solar_dir,
        'meteo_dir': meteo_dir
    }

# ---------------------------------
# Data Loading and Preparation
# ---------------------------------

def load_csv_files(directory, file_pattern=None):
    """Load multiple CSV files from a directory with optional pattern matching."""
    all_data = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist.")
        return pd.DataFrame()
    
    for filename in os.listdir(directory):
        if not filename.endswith('.csv'):
            continue
            
        if file_pattern and file_pattern not in filename:
            continue
            
        file_path = os.path.join(directory, filename)
        try:
            df = pd.read_csv(file_path)
            df['source_file'] = filename
            all_data.append(df)
            print(f"Loaded {filename}, {len(df)} rows")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not all_data:
        print("No matching files found.")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def prepare_min_data(min_df, aggregate_to_hourly=True):
    """
    Prepare inverter min data (PAC).
    Similar to original prepare_pv_data but simplified and focusing on PAC only.
    """
    if min_df.empty:
        print("Warning: Empty min dataframe.")
        return pd.DataFrame()
    
    df = min_df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Check if this is a min file with inverter data
    if 'pac' not in df.columns or 'inv' not in df.columns:
        print("Warning: Input data does not appear to be inverter data (missing PAC or INV columns).")
        return pd.DataFrame()
    
    print("Processing inverter data from min file")
    
    # Create datetime column
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour'] = df['datetime'].dt.hour
    
    # For instant power (PAC), summarize by datetime
    print("Target: Instant power (PAC)")
    df_summarized = df.groupby('datetime').agg({
        'pac': 'sum',  # Sum power across all inverters
        'status': lambda x: (x == 6).sum()  # Count running inverters
    }).reset_index()
    
    # Rename and convert if needed
    df_summarized = df_summarized.rename(columns={'pac': 'pv_output'})
    if df_summarized['pv_output'].max() > 100:
        df_summarized['pv_output'] = df_summarized['pv_output'] / 1000
        print("Converting power from W to kW")
    
    # Add hour column and set datetime as index
    df_summarized['hour'] = df_summarized['datetime'].dt.hour
    df_summarized = df_summarized.set_index('datetime').sort_index()
    
    # Aggregate to hourly resolution if requested
    if aggregate_to_hourly:
        print("Aggregating data to hourly resolution...")
        df_summarized['hour_rounded'] = df_summarized.index.floor('h')
        
        df_hourly = df_summarized.groupby('hour_rounded').agg({
            'pv_output': 'mean',
            'hour': 'first',
            'status': 'mean' if 'status' in df_summarized.columns else 'sum'
        })
        
        df_hourly.index.name = 'datetime'
        print(f"Hourly data: {df_hourly.shape[0]} records")
        return df_hourly
    
    return df_summarized

def prepare_pred_data(pred_df):
    """
    Prepare weather prediction data.
    Same logic as original function with better handling of duplicate timestamps.
    """
    if pred_df.empty:
        print("Warning: Empty prediction dataframe.")
        return pd.DataFrame()
    
    df = pred_df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    # Check required columns
    required_cols = ['date', 'time', 'pred_glob_ctrl', 'pred_t_2m_ctrl']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Prediction data missing required columns: {missing_cols}")
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
    
    # Handle duplicate timestamps (more robust approach)
    if df.index.duplicated().any():
        print(f"Found {df.index.duplicated().sum()} duplicate timestamps in prediction data")
        
        # Get unique indices
        unique_indices = ~df.index.duplicated(keep='first')
        df = df[unique_indices]
        
        print(f"After removing duplicates: {df.shape[0]} records")
    
    print(f"Prediction data prepared: {df.shape[0]} records")
    return df

def merge_datasets(pv_df, pred_df, tolerance='1h'):
    """
    Merge PV and prediction data using merge_asof for better timestamp matching.
    """
    if pv_df.empty or pred_df.empty:
        print("Warning: Cannot merge with empty dataframes.")
        return pd.DataFrame()
    
    print(f"PV data shape: {pv_df.shape}")
    print(f"Prediction data shape: {pred_df.shape}")
    
    # Reset indices for merging
    pv_reset = pv_df.reset_index()
    pred_reset = pred_df.reset_index()
    
    # Rename index column for clarity
    pv_reset = pv_reset.rename(columns={'datetime': 'pv_datetime'})
    pred_reset = pred_reset.rename(columns={'target_datetime': 'pred_datetime'})
    
    # Ensure hour column exists in both dataframes
    if 'hour' not in pv_reset.columns and 'pv_datetime' in pv_reset.columns:
        pv_reset['hour'] = pv_reset['pv_datetime'].dt.hour
    
    if 'hour' not in pred_reset.columns and 'pred_datetime' in pred_reset.columns:
        pred_reset['hour'] = pred_reset['pred_datetime'].dt.hour
    
    # For merge_asof, the "on" parameter should be dates or times
    # Create a common datetime column for merging
    pv_reset['datetime'] = pv_reset['pv_datetime']
    pred_reset['datetime'] = pred_reset['pred_datetime']
    
    # Sort values as required by merge_asof
    pv_reset = pv_reset.sort_values('datetime')
    pred_reset = pred_reset.sort_values('datetime')
    
    # Perform merge_asof
    # This is critical for maintaining the good quality of matching
    merged = pd.merge_asof(
        pv_reset,
        pred_reset,
        on='datetime',
        direction='nearest',
        tolerance=pd.Timedelta(tolerance)
    )
    
    # Ensure 'hour' column exists after merging
    if 'hour_x' in merged.columns and 'hour_y' in merged.columns:
        # If we have both hour columns from both sources, prefer the one from PV data
        merged['hour'] = merged['hour_x']
    elif 'hour_x' in merged.columns:
        merged['hour'] = merged['hour_x']
    elif 'hour_y' in merged.columns:
        merged['hour'] = merged['hour_y']
    elif 'datetime' in merged.columns:
        # If no hour column, extract from datetime
        merged['hour'] = merged['datetime'].dt.hour
    
    # Add time-based features (critical for model performance)
    merged['hour_sin'] = np.sin(2 * np.pi * merged['hour'] / 24)
    merged['hour_cos'] = np.cos(2 * np.pi * merged['hour'] / 24)
    merged['day_of_year'] = merged['datetime'].dt.dayofyear / 365.0
    
    # Filter for daylight hours (6am-7pm)
    daylight_df = merged[(merged['hour'] >= 6) & (merged['hour'] <= 19)]
    
    print(f"Merged data: {daylight_df.shape[0]} records, {daylight_df.shape[1]} columns")
    if daylight_df.shape[0] > 0:
        print(f"Columns: {daylight_df.columns.tolist()}")
    
    return daylight_df

def handle_missing_data(df):
    """
    Handle missing data by interpolation - simplified from original function.
    """
    if df.empty:
        return df
    
    missing_count = df.isna().sum()
    has_missing = missing_count.sum() > 0
    
    if has_missing:
        print("Missing values found:")
        print(missing_count[missing_count > 0])
        
        processed_df = df.copy()
        
        # Identify numeric columns (excluding datetime)
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = [col for col in processed_df.columns if 'datetime' in col.lower() or 'date' in col.lower()]
        
        for col in datetime_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
        
        # Interpolate numeric columns
        for col in numeric_cols:
            if processed_df[col].isna().any():
                processed_df[col] = processed_df[col].interpolate(method='linear').ffill().bfill()
        
        # Handle object columns if needed
        object_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
        for col in object_cols:
            if processed_df[col].isna().any() and col not in ['source_file']:
                processed_df[col] = processed_df[col].ffill().bfill()
        
        # Check if there are still missing values
        still_missing = processed_df.isna().sum()
        if still_missing.sum() > 0:
            print("Missing values after handling:")
            print(still_missing[still_missing > 0])
            
            # Drop any remaining rows with missing values in important columns
            pred_cols = [col for col in numeric_cols if col.startswith('pred_') or col in ['hour_sin', 'hour_cos', 'day_of_year']]
            processed_df = processed_df.dropna(subset=pred_cols)
            print(f"After dropping rows with missing values: {processed_df.shape[0]} records")
        
        return processed_df
    else:
        return df

def prepare_model_features_pred_only(df):
    """
    Prepare features for prediction-only model.
    Selects only prediction features - no measurements.
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
    
    # Time-based features (always include - these are very important!)
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
    
    print(f"Selected features: {features}")
    return X, y

# ---------------------------------
# Model Training and Prediction
# ---------------------------------

def train_prediction_model(X, y):
    """
    Train a RandomForest regression model for PV prediction.
    Includes GridSearchCV for hyperparameter tuning like the original.
    """
    if X.empty or y is None or y.empty:
        print("Error: Cannot train model with empty data.")
        return None
    
    print(f"Training RandomForest model with {X.shape[1]} features and {X.shape[0]} samples")
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features (important for some models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize model with default values
    model = RandomForestRegressor(random_state=42)
    
    # Parameter grid for optimization - same as original
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Perform hyperparameter tuning
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
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
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Return model information
    model_info = {
        'model': best_model,
        'scaler': scaler,
        'metrics': metrics,
        'features': X.columns.tolist(),
        'feature_importance': feature_importance,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'pred_only_model',
        'target_variable': 'pac',
        'hourly_aggregation': True
    }
    
    return model_info

def predict_pv_output(model_info, prediction_data):
    """
    Predict PV output using the trained model.
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
        
        # Try to create missing time-based features
        if 'hour_sin' in missing_features and 'hour' in prediction_data.columns:
            prediction_data['hour_sin'] = np.sin(2 * np.pi * prediction_data['hour'] / 24)
        
        if 'hour_cos' in missing_features and 'hour' in prediction_data.columns:
            prediction_data['hour_cos'] = np.cos(2 * np.pi * prediction_data['hour'] / 24)
        
        if 'day_of_year' in missing_features and 'datetime' in prediction_data.columns:
            prediction_data['day_of_year'] = prediction_data['datetime'].dt.dayofyear / 365.0
        
        # Check if any features are still missing
        still_missing = [feature for feature in features if feature not in prediction_data.columns]
        if still_missing:
            print(f"Cannot create these features: {still_missing}")
            features = [feature for feature in features if feature in prediction_data.columns]
    
    # Extract features for prediction
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

def save_model(model_info, output_path=None):
    """Save the trained model to a file."""
    if model_info is None:
        print("Error: No model to save.")
        return None
    
    # Create default output directory if needed
    if output_path is None:
        output_dir = 'models'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with current date
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = os.path.join(output_dir, f"pred_only_model_hourly_{current_date}.pkl")
    
    # Save the model
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"Model saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def save_predictions_to_csv(predictions_df, output_path=None):
    """Save prediction results to a CSV file."""
    if predictions_df is None or predictions_df.empty:
        print("Error: No predictions to save.")
        return None
    
    # Create default output directory if needed
    if output_path is None:
        output_dir = 'predictions'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with current date and time
        current_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        output_path = os.path.join(output_dir, f"pv_predictions_{current_date}.csv")
    
    # Prepare data for output
    output_df = predictions_df.copy()
    
    # Format datetime columns
    for col in ['datetime', 'pv_datetime', 'pred_datetime']:
        if col in output_df.columns and pd.api.types.is_datetime64_any_dtype(output_df[col]):
            output_df[col] = output_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save to CSV
    try:
        output_df.to_csv(output_path, index=False)
        print(f"Predictions saved to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return None

# ---------------------------------
# Main Execution
# ---------------------------------

def train_solar_pred_model(data_date=None, save_outputs=True):
    """
    Main function to train a prediction model using min and pred files only.
    Improved version that should maintain the high performance of the original.
    """
    print("Solar Energy Production Prediction Model - Improved Version")
    print("=" * 60)
    print(f"Using hourly aggregation: True")
    
    # Get directories
    dirs = get_directories(data_date)
    print(f"Using data directories:")
    for key, path in dirs.items():
        print(f"  {key}: {path}")
    
    # 1. Load and prepare min data
    print("\nLoading min files...")
    min_files = load_csv_files(dirs['solar_dir'], file_pattern='min')
    
    if min_files.empty:
        print("ERROR: No min files found. Check directory and file patterns.")
        return None
    
    min_data = prepare_min_data(min_files, aggregate_to_hourly=True)
    
    if min_data.empty:
        print("ERROR: Failed to prepare min data.")
        return None
    
    # 2. Load and prepare prediction data
    print("\nLoading prediction files...")
    pred_files = load_csv_files(dirs['meteo_dir'], file_pattern='Pred')
    
    if pred_files.empty:
        print("ERROR: No prediction files found. Check directory and file patterns.")
        return None
    
    pred_data = prepare_pred_data(pred_files)
    
    if pred_data.empty:
        print("ERROR: Failed to prepare prediction data.")
        return None
    
    # 3. Merge min and prediction data - using the improved merge_datasets function
    merged_data = merge_datasets(min_data, pred_data, tolerance='1h')
    
    if merged_data.empty:
        print("ERROR: Failed to merge min and prediction data.")
        return None
    
    # 4. Handle missing data
    clean_data = handle_missing_data(merged_data)
    
    if clean_data.empty:
        print("ERROR: No data left after handling missing values.")
        return None
    
    # 5. Prepare features for model
    X, y = prepare_model_features_pred_only(clean_data)
    
    if X.empty or y is None:
        print("ERROR: Failed to prepare model features.")
        return None
    
    # 6. Train model - using the improved training with GridSearchCV
    model_info = train_prediction_model(X, y)
    
    if model_info is None:
        print("ERROR: Model training failed.")
        return None
    
    # 7. Generate predictions on the training data
    predictions = predict_pv_output(model_info, clean_data)
    
    # 8. Visualize results
    plt.figure(figsize=(12, 6))
    plt.scatter(clean_data['pv_output'], predictions['predicted_pv_output'], alpha=0.5)
    max_val = max(clean_data['pv_output'].max(), predictions['predicted_pv_output'].max())
    plt.plot([0, max_val], [0, max_val], 'r--')
    plt.xlabel('Actual PV Output')
    plt.ylabel('Predicted PV Output')
    plt.title('Prediction Model Performance')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(model_info['feature_importance']['Feature'], 
             model_info['feature_importance']['Importance'])
    plt.xlabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # 9. Save outputs if requested
    model_path = None
    pred_path = None
    
    if save_outputs:
        # Save model
        model_path = save_model(model_info)
        
        # Save predictions
        if not predictions.empty:
            pred_path = save_predictions_to_csv(predictions)
    
    # 10. Return results
    return {
        'model_info': model_info,
        'predictions': predictions,
        'model_path': model_path,
        'pred_path': pred_path
    }

# Function for prediction using saved model
def predict_with_saved_model(model_path, pred_data):
    """
    Predict PV output using a saved model and new prediction data.
    
    Parameters:
        model_path (str): Path to the saved model file
        pred_data (DataFrame or str): Weather prediction data or path to CSV file
        
    Returns:
        DataFrame: Prediction data with added PV output predictions
    """
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model_info = pickle.load(f)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Prepare prediction data
    if isinstance(pred_data, str):
        # If pred_data is a file path
        try:
            pred_df = pd.read_csv(pred_data)
        except Exception as e:
            print(f"Error loading prediction data: {e}")
            return None
    else:
        # If pred_data is already a DataFrame
        pred_df = pred_data.copy()
    
    # Process prediction data
    pred_df.columns = [col.lower() for col in pred_df.columns]
    
    # Ensure datetime column exists
    if 'datetime' not in pred_df.columns and 'date' in pred_df.columns and 'time' in pred_df.columns:
        pred_df['datetime'] = pd.to_datetime(pred_df['date'] + ' ' + pred_df['time'])
    
    # Add hour and date columns if needed
    if 'hour' not in pred_df.columns and 'datetime' in pred_df.columns:
        pred_df['hour'] = pred_df['datetime'].dt.hour
    
    # Add required time-based features
    pred_df['hour_sin'] = np.sin(2 * np.pi * pred_df['hour'] / 24)
    pred_df['hour_cos'] = np.cos(2 * np.pi * pred_df['hour'] / 24)
    pred_df['day_of_year'] = pred_df['datetime'].dt.dayofyear / 365.0
    
    # Make predictions
    predictions = predict_pv_output(model_info, pred_df)
    
    # Keep only essential columns for output
    if 'predicted_pv_output' in predictions.columns:
        output_cols = ['datetime', 'hour', 'predicted_pv_output']
        output_cols.extend([col for col in predictions.columns if col.startswith('pred_')])
        result_df = predictions[output_cols].copy()
        return result_df
    else:
        return predictions
    
if __name__ == "__main__":
    # Date des données à utiliser pour l'entraînement
    data_date = '2025-03-14'  # Changez la date selon vos besoins
    
    # Entraîner le modèle
    results = train_solar_pred_model(data_date=data_date, save_outputs=True)
    
    if results:
        print("=" * 60)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
        print(f"Model metrics:")
        for metric, value in results['model_info']['metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        if results['model_path']:
            print(f"Model saved to: {results['model_path']}")
        
        if results['pred_path']:
            print(f"Predictions saved to: {results['pred_path']}")
        
        # Afficher les caractéristiques les plus importantes
        importance_df = results['model_info']['feature_importance']
        print("\nTop feature importance:")
        print(importance_df.head(3))
        
        # Tester le modèle sur des données de prédiction futures
        print("\nSimulating prediction on future data...")
        
        # Exemple - utilisez votre propre fichier de prédiction si disponible
        # future_preds = predict_with_saved_model(results['model_path'], 'path/to/future_pred_data.csv')
        
        print("\nReady for deployment to API.")
    else:
        print("=" * 60)
        print("MODEL TRAINING FAILED")
        print("=" * 60)
        print("Check error messages above for troubleshooting.")