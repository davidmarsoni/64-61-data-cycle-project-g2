"""
Shared utility functions for data cleaning operations
"""
import os
import chardet
import traceback
import logging
import pandas as pd
from functools import lru_cache
from datetime import datetime

# Global error log for tracking errors across modules
error_log = []

def log_error(phase, error_msg, trace=None):
    """Log an error message with optional traceback.

    Args:
        phase: The phase of execution where the error occurred
        error_msg: The error message
        trace: Traceback information. Defaults to None.
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

@lru_cache(maxsize=100)
def detect_encoding(file_path, num_bytes=10000):
    """Detect the encoding of a file using chardet.

    Args:
        file_path: path to the file either relative or absolute
        num_bytes (int, optional): number of bytes to read for detection. Defaults to 10000.

    Returns:
        detected encoding
    """
    with open(file_path, 'rb') as f:
        rawdata = f.read(num_bytes)
    result = chardet.detect(rawdata)
    return result['encoding']

def normalize_date_format(date_series, log_prefix=""):
    """
    Normalize dates in a pandas Series to YYYY-MM-DD format.
    
    Args:
        date_series: Pandas Series containing date strings or timestamps
        log_prefix: Optional prefix for log messages
        
    Returns:
        Pandas Series with normalized date strings
    """
    try:
        # Check if dates are in DD.MM.YYYY format
        sample_date = date_series.iloc[0]
        if '.' in str(sample_date) and str(sample_date).count('.') == 2:
            # For format like "21.02.23" or "21.02.2023"
            if len(str(sample_date).split('.')[-1]) == 2:  # Short year format (YY)
                date_series = pd.to_datetime(date_series, format='%d.%m.%y', errors='coerce')
            else:  # Full year format (YYYY)
                date_series = pd.to_datetime(date_series, format='%d.%m.%Y', errors='coerce')
        else:
            # Fallback to automatic format detection
            date_series = pd.to_datetime(date_series, errors='coerce')
        
        # Convert to standard YYYY-MM-DD format
        return date_series.dt.strftime('%Y-%m-%d')
        
    except Exception as e:
        logging.error(f"{log_prefix} Date conversion error: {str(e)}")
        # Try European format with dayfirst
        date_series = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
        return date_series.dt.strftime('%Y-%m-%d')

def normalize_time_format(time_series, log_prefix=""):
    """
    Normalize times in a pandas Series to HH:MM:SS format.
    
    Args:
        time_series: Pandas Series containing time strings or timestamps
        log_prefix: Optional prefix for log messages
        
    Returns:
        Pandas Series with normalized time strings
    """
    try:
        # First ensure time is in string format for proper parsing
        time_series = time_series.astype(str)
        # Handle HH:MM:SS format
        time_series = pd.to_datetime(time_series, format='%H:%M:%S', errors='coerce')
        return time_series.dt.strftime('%H:%M:%S')
    except Exception as e:
        logging.error(f"{log_prefix} Time conversion error: {str(e)}")
        # Try with more flexible parsing
        time_series = pd.to_datetime(time_series, errors='coerce')
        return time_series.dt.strftime('%H:%M:%S')

def get_date_obj_from_str(date_str):
    """
    Convert a date string to a datetime object
    
    Args:
        date_str: String date in YYYY-MM-DD format
        
    Returns:
        datetime object
    """
    return datetime.strptime(date_str, '%Y-%m-%d')

def get_time_obj_from_str(time_str):
    """
    Convert a time string to a datetime object
    
    Args:
        time_str: String time in HH:MM:SS format
        
    Returns:
        datetime object
    """
    return datetime.strptime(time_str, '%H:%M:%S')
