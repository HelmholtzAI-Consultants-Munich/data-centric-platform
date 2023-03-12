import sys
import warnings

from PyQt5.QtWidgets import QApplication

from dcp_client.app import Application
from dcp_client.bento_mlclient import BentomMLClient
from dcp_client.gui.welcome_window import WelcomeWindow

warnings.simplefilter('ignore')

def main():
    app = Application(
        mlclient=BentomMLClient(),
        server_ip='0.0.0.0',
        server_port=7010,
        img_filename=None, 
        eval_data_path='', 
        train_data_path='',
        inprogr_data_path='',
    )
    event_loop = QApplication(sys.argv)
    window = WelcomeWindow(app=app)
    sys.exit(event_loop.exec())


if __name__ == "__main__":
    main()