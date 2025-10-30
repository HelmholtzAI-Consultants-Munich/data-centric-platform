import argparse
import sys
import warnings
from os import path
import os
from contextlib import contextmanager

from dcp_client.app import Application
from dcp_client.gui.welcome_window import WelcomeWindow
from dcp_client.utils import settings
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync
from dcp_client.utils.utils import read_config
from PyQt5.QtWidgets import QApplication

warnings.simplefilter("ignore")

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

    settings.init()

    dir_name = path.dirname(path.abspath(__file__))

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

    if args.mode == "local":
        server_config = read_config(
            "server", config_path=path.join(dir_name, "config.yaml")
        )
    elif args.mode == "remote":
        server_config = read_config(
            "server", config_path=path.join(dir_name, args.config_remote)
        )

    # Conditional validation
    if args.multi_class and args.num_classes is None:
        parser.error("--num-classes is required when --multi-class is set. Please provide the number of classes in your data.")
    
    if not args.multi_class: 
        num_classes = 1
    else:
        num_classes = args.num_classes

    image_storage = FilesystemImageStorage()
    ml_model = BentomlModel()
    data_sync = DataRSync(
        user_name=server_config["user"],
        host_name=server_config["host"],
        server_repo_path=server_config["data-path"],
    )
    welcome_app = Application(
        ml_model=ml_model,
        num_classes=num_classes,
        syncer=data_sync,
        image_storage=image_storage,
        server_ip=server_config["ip"],
        server_port=server_config["port"],
    )
    app = QApplication(sys.argv)
    window = WelcomeWindow(welcome_app)
    with suppress_tiff_warnings(): sys.exit(app.exec())


if __name__ == "__main__":
    main()
