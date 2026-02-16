import argparse
import sys
import warnings
from os import path
import os
from contextlib import contextmanager
import logging

from dcp_client.app import Application
from dcp_client.gui.welcome_window import WelcomeWindow
from dcp_client.utils import settings
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.utils import read_config
from dcp_client.utils.logger import setup_logger, get_logger
from PyQt5.QtWidgets import QApplication

warnings.simplefilter("ignore")

# Initialize logger
logger = get_logger(__name__)

@contextmanager
def suppress_tiff_warnings():
    """Suppress only libtiff TIFFReadDirectory warnings while keeping other stderr output."""
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)

    # Create a pipe to capture stderr
    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, original_stderr_fd)
    os.close(w_fd)

    try:
        yield
    finally:
        # Restore original stderr
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stderr_fd)

        # Read captured output
        captured_output = os.read(r_fd, 10000).decode()  # adjust buffer size if needed
        os.close(r_fd)

        # Filter out TIFF warnings, print the rest
        for line in captured_output.splitlines():
            if "TIFFReadDirectory" not in line:
                sys.stderr.write(line + "\n")

def main():
    # Parse arguments first to get log level
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        choices=["local", "remote"],
        required=True,
        help="Choose mode: local or remote",
    )
    parser.add_argument(
        "-cr",
        "--config_remote",
        required=False,
        default = 'config_remote.yaml',
        help="Pass the remote config file for the project",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    # Multi-class flag
    parser.add_argument(
        "--multi-class",
        action="store_true",
        help="Enable multi-class classification",
    )

    # Num-classes (conditionally required)
    parser.add_argument(
        "--num-classes",
        type=int,
        help="Number of classes (required if --multi-class is set)",
    )

    args = parser.parse_args()
    
    # Convert log level string to logging constant
    log_level = getattr(logging, args.log_level)
    
    # Set up logging with optional file logging
    setup_logger(log_level=log_level, log_file=os.path.join(os.path.expanduser("~"), ".dcp_client", "dcp_client.log"))
    
    logger.info("Starting DCP Client...")
    logger.debug(f"Log level set to: {args.log_level}")

    settings.init()

    dir_name = path.dirname(path.abspath(__file__))
    logger.debug(f"Client directory: {dir_name}")
    logger.info(f"Mode: {args.mode}")

    try:
        if args.mode == "local":
            server_config = read_config(
                "server", config_path=path.join(dir_name, "config.yaml")
            )
            logger.debug("Loaded local server config")
        elif args.mode == "remote":
            server_config = read_config(
                "server", config_path=path.join(dir_name, args.config_remote)
            )
            logger.debug(f"Loaded remote server config from {args.config_remote}")

        # Conditional validation
        if args.multi_class and args.num_classes is None:
            logger.error("--num-classes is required when --multi-class is set")
            parser.error("--num-classes is required when --multi-class is set. Please provide the number of classes in your data.")
        
        if not args.multi_class: 
            num_classes = 1
        else:
            num_classes = args.num_classes
        
        logger.info(f"Number of classes: {num_classes}, Multi-class mode: {args.multi_class}")

        logger.debug("Initializing image storage...")
        image_storage = FilesystemImageStorage()
        
        logger.debug("Initializing ML model...")
        ml_model = BentomlModel()
        
        logger.debug("Initializing application...")
        welcome_app = Application(
            ml_model=ml_model,
            num_classes=num_classes,
            image_storage=image_storage,
            server_ip=server_config["ip"],
            server_port=server_config["port"],
        )
        
        logger.info(f"Connecting to server at {server_config['ip']}:{server_config['port']}")
        app = QApplication(sys.argv)
        window = WelcomeWindow(welcome_app)
        logger.info("Launching UI...")
        with suppress_tiff_warnings(): sys.exit(app.exec())
    
    except Exception as e:
        logger.exception(f"An error occurred during initialization: {e}")
        raise



if __name__ == "__main__":
    main()
