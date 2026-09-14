import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name: str) -> logging.Logger:
    """Sets up a structured, enterprise-grade logger."""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Only configure if no handlers exist to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Log format
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        # File Handler (Rotating, max 5MB per file, keep 3 backups)
        file_handler = RotatingFileHandler(
            filename="logs/system.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
    return logger
