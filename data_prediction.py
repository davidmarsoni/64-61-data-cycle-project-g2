"""
Main entry point for all the prediction processes.
This script runs both energy production and consumption prediction models.
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add the path to the machineLearning directory to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'machineLearning'))

# Import prediction modules
from machineLearning.EnergyProductionPrediction import main_compare_models
from machineLearning.EnergyConsumption import run_improved_hourly_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'predictions/prediction_log_{datetime.now().strftime("%Y%m%d")}.log')
    ]
)
logger = logging.getLogger(__name__)

def run_energy_production_prediction(data_date=None, save_outputs=True, target_variable='pac'):
    """
    Run the energy production prediction model
    
    Parameters:
        data_date (str): Date string in 'YYYY-MM-DD' format
        save_outputs (bool): Whether to save outputs to files
        target_variable (str): Target variable to predict - 'pac' for power or 'daysum' for daily energy
    
    Returns:
        bool: Success status
    """
    logger.info("Starting Energy Production Prediction")
    try:
        results = main_compare_models(
            data_date=data_date,
            save_outputs=save_outputs,
            target_variable=target_variable
        )
        logger.info("Energy Production Prediction completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Energy Production Prediction: {str(e)}")
        return False

def run_energy_consumption_prediction(data_date=None, model_type='xgboost', save_outputs=True):
    """
    Run the energy consumption prediction model
    
    Parameters:
        data_date (str): Date string in 'YYYY-MM-DD' format
        model_type (str): Type of model to use
        save_outputs (bool): Whether to save outputs to files
    
    Returns:
        bool: Success status
    """
    logger.info("Starting Energy Consumption Prediction")
    try:
        results = run_improved_hourly_model(
            data_date=data_date,
            model_type=model_type,
            save_outputs=save_outputs
        )
        logger.info("Energy Consumption Prediction completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error in Energy Consumption Prediction: {str(e)}")
        return False

def run_all_predictions(data_date=None, save_outputs=True, energy_model_type='xgboost'):
    """Run all prediction processes in sequence"""
    logger.info(f"Running all prediction models with date={data_date}")
    
    success = True
    
    try:
        # Run energy production prediction
        prod_success = run_energy_production_prediction(
            data_date=data_date,
            save_outputs=save_outputs
        )
        
        # Run energy consumption prediction
        cons_success = run_energy_consumption_prediction(
            data_date=data_date,
            model_type=energy_model_type,
            save_outputs=save_outputs
        )
        
        success = prod_success and cons_success
        return success
        
    except Exception as e:
        logger.error(f"Error running predictions: {str(e)}")
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run energy prediction models')
    parser.add_argument('--date', type=str, default=None,
                        help='Data date in YYYY-MM-DD format (default: current date)')
    parser.add_argument('--model', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'randomforest', 'logisticregression'],
                        help='Energy consumption model type (default: xgboost)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results (default: True)')
    parser.add_argument('--no-save', dest='save', action='store_false',
                        help='Do not save results')
    parser.add_argument('--production-only', action='store_true',
                        help='Run only production prediction')
    parser.add_argument('--consumption-only', action='store_true',
                        help='Run only consumption prediction')
    
    args = parser.parse_args()
    
    success = True
    
    if args.production_only:
        success = run_energy_production_prediction(args.date, args.save)
    elif args.consumption_only:
        success = run_energy_consumption_prediction(args.date, args.model, args.save)
    else:
        success = run_all_predictions(args.date, args.save, args.model)
    
    exit(0 if success else 1)