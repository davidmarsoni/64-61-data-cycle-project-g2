"""
Shared utility functions for data cleaning operations
"""
import os
import chardet
import traceback
import logging
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
