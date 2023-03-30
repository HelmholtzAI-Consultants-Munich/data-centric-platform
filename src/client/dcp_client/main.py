import sys
from PyQt5.QtWidgets import QApplication
from welcome_window import WelcomeWindow
from app import Application
from bentoml_model import BentomlModel

import warnings
warnings.simplefilter('ignore')

import settings
settings.init()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    bentoml_model = BentomlModel()
    welcome_app = Application(bentoml_model)
    window = WelcomeWindow(welcome_app)
    sys.exit(app.exec())