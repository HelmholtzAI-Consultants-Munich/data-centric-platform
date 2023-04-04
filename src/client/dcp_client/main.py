import sys
from PyQt5.QtWidgets import QApplication
from welcome_window import WelcomeWindow
from app import Application
from bentoml_model import BentomlModel
from fsimagestorage import FilesystemImageStorage

import warnings
warnings.simplefilter('ignore')

import settings
settings.init()

if __name__ == "__main__":
    image_storage = FilesystemImageStorage()
    ml_model = BentomlModel()
    welcome_app = Application(ml_model=ml_model, image_storage=image_storage)
    app = QApplication(sys.argv)
    window = WelcomeWindow(welcome_app)
    sys.exit(app.exec())