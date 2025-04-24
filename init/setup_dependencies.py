"""
Setup script to ensure all dependencies are installed before running the data pipeline.
"""
import sys
import os
import logging

# Add parent directory to sys.path so we can import the config module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Set up basic console logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Import and use the ensure_installed function from config.py
try:
    from config import ensure_installed
    
    # Install required packages
    ensure_installed('pandas')
    ensure_installed('chardet')
    ensure_installed('python-dotenv')
    ensure_installed('pyodbc')  # For database connections in ETL
    ensure_installed('keyring')  # For credential management
    
    logging.info("All dependencies are installed successfully!")
    sys.exit(0)
except Exception as e:
    logging.error(f"Failed to install dependencies: {e}")
    sys.exit(1)