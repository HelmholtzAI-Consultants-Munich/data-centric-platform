import sys
from PyQt5.QtWidgets import QApplication
from welcome_window import WelcomeWindow
from app import Application
from fsimagestorage import FilesystemImageStorage

import warnings
warnings.simplefilter('ignore')

import settings
settings.init()

if __name__ == "__main__":
    image_storage = FilesystemImageStorage()
    app = QApplication(sys.argv)
    welcome_app = Application(image_storage=image_storage)
    window = WelcomeWindow(welcome_app)
    sys.exit(app.exec())