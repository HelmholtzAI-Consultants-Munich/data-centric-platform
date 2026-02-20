from os import path
import sys
import subprocess
import logging
import logging.handlers
import os

from dcp_server.utils.helpers import read_config
from dcp_server.utils.logger import setup_logger, get_logger

# Initialize logger
logger = get_logger(__name__)


def configure_root_logger() -> None:
    """Configure root logger to capture all logging including BentoML's."""
    log_file = os.path.join(os.path.expanduser("~"), ".dcp_server", "dcp_server.log")
    log_path = os.path.dirname(log_file)
    os.makedirs(log_path, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def main() -> None:
    """
    Contains main functionality related to the server.
    """
    # Configure root logger BEFORE BentoML starts
    configure_root_logger()
    
    logger.info("Starting DCP Server...")

    # global config_path
    # args = sys.argv
    # if len(args) > 1:
    #     if path.exists(args[1]) and args[1].endswith('.cfg'):
    #         config_path = args[1]
    # else:
    #     config_path = 'config.cfg'

    try:
        local_path = path.join(path.dirname(path.realpath(__file__)))
        logger.debug(f"Server directory: {local_path}")
        
        dir_name = path.dirname(path.abspath(sys.argv[0]))
        logger.debug(f"Working directory: {dir_name}")
        
        logger.debug("Loading service configuration...")
        service_config = read_config(
            "service", config_path=path.join(dir_name, "config.yaml")
        )
        service_name = str(service_config["service_name"])
        port = str(service_config["port"])
        host = str(service_config["host"])
        timeout = str(service_config["timeout"])
        
        logger.info(f"Service: {service_name}, Port: {port}, Host: {host}, Timeout: {timeout}")
        logger.info("Starting BentoML server...")
        
        subprocess.run(
            [
                "bentoml",
                "serve",
                "service:"+service_name,
                "--working-dir",
                local_path,
                "--host",
                host,
                "--port=" + port,
                "--timeout=" + timeout
            ]
        )
        logger.info("Server shutdown")
    
    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        raise
 


if __name__ == "__main__":
    main()
