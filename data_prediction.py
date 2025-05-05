"""
Dedicated prediction workflow script.
This script:
1. Trains prediction models (consumption and production)
2. Calls APIs to get predictions
3. Loads prediction data into the database via ETL

All paths are properly configured to ensure consistency across components.
"""
import os
import sys
import logging
import argparse
from datetime import datetime
import time

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

# Configure logging
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, f'prediction_pipeline_{datetime.now().strftime("%Y%m%d")}.log'))
    ]
)
logger = logging.getLogger("PredictionPipeline")

# Import model training modules
try:
    from machineLearning.EnergyConsumption import run_improved_hourly_model
    from machineLearning.EnergyProductionPrediction import solar_prediction_pipeline
    
    # Import API client
    from machineLearning.getPredictions import call_prediction_api
    
    # Import FactPrediction ETL
    from ETL.Fact.ETL_process_FactPrediction import populate_dim_tables_and_facts as run_prediction_etl
    
    from ETL.utils.logging_utils import setup_logging, send_error_summary
    
    modules_loaded = True
except ImportError as e:
    logger.error(f"Failed to import required modules: {str(e)}", exc_info=True)
    modules_loaded = False

# Create required directories
def ensure_directories_exist():
    """Create all necessary directories for the prediction pipeline"""
    # API results directory - where prediction CSVs are stored
    api_results_dir = os.path.join(project_root, 'machineLearning', 'api_results')
    os.makedirs(api_results_dir, exist_ok=True)
    logger.info(f"API results directory: {api_results_dir}")
    
    # Model directory
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Models directory: {model_dir}")
    
    # Today's meteo dir for API
    today = datetime.now().strftime('%Y-%m-%d')
    meteo_dir = os.path.join(os.getenv('BASE_DIR', 'C:/DataCollection'), f'cleaned_data_{today}', 'Meteo')
    os.makedirs(meteo_dir, exist_ok=True)
    logger.info(f"Meteo directory: {meteo_dir}")
    
    return {
        'api_results_dir': api_results_dir,
        'model_dir': model_dir,
        'meteo_dir': meteo_dir
    }

def run_consumption_model(date=None, model_type='xgboost', save_outputs=True):
    """Run the energy consumption prediction model"""
    logger.info("Starting Energy Consumption Prediction training")
    try:
        results = run_improved_hourly_model(
            data_date=date,
            model_type=model_type,
            save_outputs=save_outputs
        )
        logger.info("Energy Consumption Prediction training completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Energy Consumption Prediction: {str(e)}", exc_info=True)
        return False

def run_production_model(date=None, save_outputs=True, target_variable='pac'):
    """Run the energy production prediction model"""
    logger.info("Starting Energy Production Prediction training")
    try:
        predictions_path = solar_prediction_pipeline(
            data_date=date,
            save_outputs=save_outputs,
            target_variable=target_variable,
            training_cutoff_days=3
        )
        if predictions_path:
            logger.info(f"Energy Production Prediction training completed successfully: {predictions_path}")
            return True
        else:
            logger.warning("Energy Production Prediction training completed but no output path returned")
            return False
    except Exception as e:
        logger.error(f"Error in Energy Production Prediction: {str(e)}", exc_info=True)
        return False

def run_prediction_pipeline(date=None, model_type='xgboost', skip_training=False, 
                           skip_api_calls=False, skip_etl=False, api_retries=3,
                           force_processing=False):
    """
    Run the complete prediction pipeline
    
    Parameters:
        date (str): Date string in YYYY-MM-DD format (default: current date)
        model_type (str): Type of consumption model to use
        skip_training (bool): Skip model training steps
        skip_api_calls (bool): Skip API calls to retrieve predictions
        skip_etl (bool): Skip ETL process for storing predictions
        api_retries (int): Number of times to retry API calls if they fail
        force_processing (bool): Continue even if some steps fail
    
    Returns:
        bool: True if all steps succeeded, False otherwise
    """
    logger.info("="*50)
    logger.info("STARTING PREDICTION PIPELINE")
    logger.info("="*50)
    
    # Create all necessary directories
    directories = ensure_directories_exist()
    
    all_steps_successful = True
    
    # Step 1: Train prediction models (if not skipped)
    if not skip_training:
        logger.info("-"*50)
        logger.info("STEP 1: TRAINING PREDICTION MODELS")
        logger.info("-"*50)
        
        # Run consumption model
        consumption_success = run_consumption_model(date, model_type, True)
        if not consumption_success:
            logger.error("Consumption model training failed")
            all_steps_successful = False
            if not force_processing:
                return False
                
        # Run production model
        production_success = run_production_model(date, True, 'pac')
        if not production_success:
            logger.error("Production model training failed")
            all_steps_successful = False
            if not force_processing:
                return False
    else:
        logger.info("Skipping model training step as requested")
    
    # Step 2: Call prediction APIs (if not skipped)
    if not skip_api_calls:
        logger.info("-"*50)
        logger.info("STEP 2: CALLING PREDICTION APIS")
        logger.info("-"*50)
        
        # Retry mechanism for API calls
        api_success = False
        for attempt in range(api_retries):
            logger.info(f"API call attempt {attempt+1}/{api_retries}")
            
            # Wait a bit before retrying
            if attempt > 0:
                logger.info(f"Waiting 5 seconds before retry...")
                time.sleep(5)
            
            try:
                call_prediction_api()
                api_success = True
                logger.info("Successfully called prediction APIs")
                break
            except Exception as e:
                logger.error(f"Error calling prediction APIs (attempt {attempt+1}): {str(e)}", exc_info=True)
        
        if not api_success:
            logger.error(f"API calls failed after {api_retries} attempts")
            all_steps_successful = False
            if not force_processing:
                return False
    else:
        logger.info("Skipping API calls step as requested")
    
    # Step 3: Run the ETL process to load predictions into the database
    if not skip_etl:
        logger.info("-"*50)
        logger.info("STEP 3: RUNNING PREDICTION ETL")
        logger.info("-"*50)
        
        try:
            setup_logging("Prediction_ETL")
            run_prediction_etl()
            logger.info("Prediction ETL process completed successfully")
        except Exception as e:
            logger.error(f"Error in Prediction ETL process: {str(e)}", exc_info=True)
            all_steps_successful = False
            if not force_processing:
                return False
    else:
        logger.info("Skipping ETL step as requested")
    
    if all_steps_successful:
        logger.info("="*50)
        logger.info("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
    else:
        logger.info("="*50)
        logger.info("PREDICTION PIPELINE COMPLETED WITH ERRORS")
        logger.info("="*50)
    
    return all_steps_successful

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run energy prediction pipeline')
    parser.add_argument('--date', type=str, default=None,
                        help='Data date in YYYY-MM-DD format (default: current date)')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'randomforest'],
                        help='Energy consumption model type (default: xgboost)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip model training steps')
    parser.add_argument('--skip-api-calls', action='store_true',
                        help='Skip API calls to retrieve predictions')
    parser.add_argument('--skip-etl', action='store_true',
                        help='Skip ETL process for storing predictions')
    parser.add_argument('--retries', type=int, default=3,
                        help='Number of API call retries (default: 3)')
    parser.add_argument('--force', action='store_true',
                        help='Continue pipeline even if some steps fail')
    
    args = parser.parse_args()
    
    logger.info("Configuration:")
    logger.info(f"  Date: {args.date if args.date else 'Current date'}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Skip training: {args.skip_training}")
    logger.info(f"  Skip API calls: {args.skip_api_calls}")
    logger.info(f"  Skip ETL: {args.skip_etl}")
    logger.info(f"  API retries: {args.retries}")
    logger.info(f"  Force processing: {args.force}")
    
    success = run_prediction_pipeline(
        date=args.date,
        model_type=args.model,
        skip_training=args.skip_training,
        skip_api_calls=args.skip_api_calls,
        skip_etl=args.skip_etl,
        api_retries=args.retries,
        force_processing=args.force
    )
    
    sys.exit(0 if success else 1)