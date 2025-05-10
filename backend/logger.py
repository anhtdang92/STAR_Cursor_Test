import logging
import os
from logging.handlers import RotatingFileHandler
import sys
from datetime import datetime

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors and timestamps"""
    
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y-%m-%d %H:%M:%S')
        return formatter.format(record)

def setup_logger(name, log_file=None, level=logging.DEBUG):
    """Set up logger with console and file handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create logs directory if it doesn't exist
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)
    
    # File Handler with rotation
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
    
    return logger

# Create main application logger
app_logger = setup_logger(
    'star_diffusion',
    log_file='backend/logs/app.log'
)

# Create CUDA-specific logger
cuda_logger = setup_logger(
    'star_diffusion.cuda',
    log_file='backend/logs/cuda.log'
)

# Create model-specific logger
model_logger = setup_logger(
    'star_diffusion.model',
    log_file='backend/logs/model.log'
)

def log_error(error, context=None):
    """Log error with context information"""
    error_msg = f"Error: {str(error)}"
    if context:
        error_msg += f" | Context: {context}"
    app_logger.error(error_msg, exc_info=True)

def log_cuda_info():
    """Log CUDA device information"""
    try:
        import torch
        if torch.cuda.is_available():
            cuda_logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            cuda_logger.info(f"CUDA Version: {torch.version.cuda}")
            cuda_logger.info(f"Available GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            cuda_logger.warning("CUDA is not available")
    except Exception as e:
        cuda_logger.error(f"Failed to get CUDA information: {str(e)}")

def log_model_info(model):
    """Log model information"""
    try:
        model_logger.info(f"Model Type: {type(model).__name__}")
        model_logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters())}")
        model_logger.info(f"Model Device: {next(model.parameters()).device}")
    except Exception as e:
        model_logger.error(f"Failed to get model information: {str(e)}") 