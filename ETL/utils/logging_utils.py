import logging
import traceback
import os
from datetime import datetime
from config import Config

# Track errors throughout the execution
error_log = []

def log_error(phase, error_msg, trace=None):
    """Log an error message with optional traceback.

    Args:
        phase (str): The phase of execution where the error occurred
        error_msg (str): The error message
        trace (str, optional): Traceback information. Defaults to None.
    """
    error_entry = {
        'phase': phase,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'message': str(error_msg),
        'traceback': trace
    }
    error_log.append(error_entry)
    if trace:
        logging.error(f"{phase}: {error_msg}")
        logging.debug(trace)
    else:
        logging.error(f"{phase}: {error_msg}")

def send_error_summary(module_name="ETL Process"):
    """Send an email summarizing all errors encountered during execution
    
    Args:
        module_name (str, optional): Name of the module for the email subject. Defaults to "ETL Process".
    """
    if not error_log:
        return
    
    # Prepare error message for email
    summary_msg = f"The following errors occurred during {module_name}:\n\n"
    for error in error_log:
        summary_msg += f"- Phase: {error['phase']}\n"
        summary_msg += f"  Time: {error['time']}\n"
        summary_msg += f"  Message: {error['message']}\n\n"
    
    # Get the first error's traceback for detailed information
    first_traceback = error_log[0].get('traceback', None) if error_log else None
    
    # Send the email and check the result
    success = Config.send_error_email(
        module_name=module_name,
        subject=f"{module_name} Errors ({len(error_log)} issues)",
        error_message=summary_msg,
        traceback_info=first_traceback
    )
    
    # Only log success if the email was actually sent
    if success:
        logging.info("Error summary email sent successfully.")

def setup_logging(module_name="ETL"):
    """Configure logging for the ETL process
    
    Args:
        module_name (str, optional): Name of the module for log identification. Defaults to "ETL".
    """
    # Create a more structured log directory path
    base_log_dir = Config.LOG_DIR
    etl_log_dir = os.path.join(base_log_dir, "ETL")
    
    # Create the ETL log directory if it doesn't exist
    os.makedirs(etl_log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_file = f"{module_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(etl_log_dir, log_file)
    
    # Clear any existing handlers first to prevent duplicate logging
    root_logger = logging.getLogger()
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create log formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"{module_name} logging initialized")
    return log_path


