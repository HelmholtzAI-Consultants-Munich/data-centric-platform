import argparse
import sys
import warnings
from os import path

from dcp_client.app import Application
from dcp_client.gui.welcome_window import WelcomeWindow
from dcp_client.utils import settings
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync
from dcp_client.utils.utils import read_config
from PyQt5.QtWidgets import QApplication

warnings.simplefilter("ignore")


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
    args = parser.parse_args()

    if args.mode == "local":
        server_config = read_config(
            "server", config_path=path.join(dir_name, "config.yaml")
        )
    elif args.mode == "remote":
        server_config = read_config(
            "server", config_path=path.join(dir_name, "config_remote.yaml")
        )

    image_storage = FilesystemImageStorage()
    ml_model = BentomlModel()
    data_sync = DataRSync(
        user_name=server_config["user"],
        host_name=server_config["host"],
        server_repo_path=server_config["data-path"],
    )
    welcome_app = Application(
        ml_model=ml_model,
        syncer=data_sync,
        image_storage=image_storage,
        server_ip=server_config["ip"],
        server_port=server_config["port"],
    )
    app = QApplication(sys.argv)
    window = WelcomeWindow(welcome_app)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
