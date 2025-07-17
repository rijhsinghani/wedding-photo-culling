"""
Comprehensive error handling and logging utilities.
"""

import os
import sys
import logging
import traceback
from functools import wraps
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
def setup_logging(log_level="INFO"):
    """Set up comprehensive logging configuration."""
    log_file = LOG_DIR / f"culling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

# Error handling decorator
def handle_errors(func):
    """Decorator to handle errors gracefully."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            logger.debug(traceback.format_exc())
            # Re-raise the exception after logging
            raise
    return wrapper

# Specific error handlers
def handle_file_error(file_path, operation="process"):
    """Handle file-related errors with context."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except FileNotFoundError:
                logger = logging.getLogger(func.__module__)
                logger.error(f"File not found: {file_path}")
                return None
            except PermissionError:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Permission denied for {operation} on: {file_path}")
                return None
            except Exception as e:
                logger = logging.getLogger(func.__module__)
                logger.error(f"Error during {operation} of {file_path}: {str(e)}")
                return None
        return wrapper
    return decorator

def log_api_error(api_name, error):
    """Log API-specific errors with context."""
    logger = logging.getLogger(__name__)
    if "quota" in str(error).lower():
        logger.warning(f"{api_name} API quota exceeded. Consider adding delays or upgrading your plan.")
    elif "key" in str(error).lower():
        logger.error(f"{api_name} API key issue. Please check your .env file.")
    else:
        logger.error(f"{api_name} API error: {str(error)}")

def validate_directory(directory_path, create_if_missing=False):
    """Validate directory exists and is accessible."""
    path = Path(directory_path)
    
    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created directory: {directory_path}")
                return True
            except Exception as e:
                logging.error(f"Failed to create directory {directory_path}: {str(e)}")
                return False
        else:
            logging.error(f"Directory does not exist: {directory_path}")
            return False
    
    if not path.is_dir():
        logging.error(f"Path is not a directory: {directory_path}")
        return False
    
    if not os.access(path, os.R_OK | os.W_OK):
        logging.error(f"Insufficient permissions for directory: {directory_path}")
        return False
    
    return True

def safe_file_operation(operation, file_path, *args, **kwargs):
    """Safely perform file operations with error handling."""
    try:
        return operation(file_path, *args, **kwargs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"File operation failed on {file_path}: {str(e)}")
        return None