import sys
from PyQt5.QtWidgets import QApplication
from .welcome_window import WelcomeWindow

import warnings
warnings.simplefilter('ignore')


def main():
    import settings
    settings.init()
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
    