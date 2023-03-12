import sys
import warnings

from PyQt5.QtWidgets import QApplication

from .app import Application
from .mlclient import MLClient
from .welcome_window import WelcomeWindow

warnings.simplefilter('ignore')



if __name__ == "__main__":
    app = Application(
        mlclient=MLClient(),
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