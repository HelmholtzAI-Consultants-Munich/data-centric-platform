"""Logging configuration for DCP Server."""

import logging
import logging.handlers
import sys
from pathlib import Path


def setup_logger(name: str = "dcp_server", log_level: int = logging.INFO, log_file: str = None) -> logging.Logger:
    """
    Set up and return a configured logger for the DCP server.
    
    :param name: Logger name (default: 'dcp_server')
    :param log_level: Logging level (default: logging.INFO)
    :param log_file: Optional file path to write logs. If None, logs only go to console
    :return: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Prevent adding multiple handlers if logger is reconfigured
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(log_level)
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name. Use this in modules to get a child logger.
    
    :param name: Logger name (typically module name via __name__)
    :return: Logger instance
    """
    return logging.getLogger(f"dcp_server.{name}")
