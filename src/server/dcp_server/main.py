from os import path
import sys
import subprocess
import logging
import os

from dcp_server.utils.helpers import read_config
from dcp_server.utils.logger import setup_logger, get_logger

# Initialize logger
logger = get_logger(__name__)



def main() -> None:
    """
    Contains main functionality related to the server.
    """
    # Set up logging with optional file logging
    setup_logger(log_file=os.path.join(os.path.expanduser("~"), ".dcp_server", "dcp_server.log"))
    
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
        timeout = str(service_config["timeout"])
        
        logger.info(f"Service: {service_name}, Port: {port}, Timeout: {timeout}")
        logger.info("Starting BentoML server...")
        
        subprocess.run(
            [
                "bentoml",
                "serve",
                "service:"+service_name,
                "--working-dir",
                local_path,
                "--reload",
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
