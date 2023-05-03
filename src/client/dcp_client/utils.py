from PyQt5.QtWidgets import  QFileIconProvider, QMessageBox
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon

from pathlib import Path, PurePath

from dcp_client import settings

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()
        self.ICON_SIZE = QSize(512,512)

    def icon(self, type: 'QFileIconProvider.IconType'):

        fn = type.filePath()

        if fn.endswith(settings.accepted_types):
            a = QPixmap(self.ICON_SIZE)
            a.load(fn)
            return QIcon(a)
        else:
            return super().icon(type)

def create_warning_box(message_text, message_title="Warning"):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(message_text)
    msg.setWindowTitle(message_title)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()

def get_relative_path(filepath): return PurePath(filepath).name

def get_path_stem(filepath): return str(Path(filepath).stem)

def get_path_name(filepath): return str(Path(filepath).name)

def get_path_parent(filepath): return str(Path(filepath).parent)

def join_path(root_dir, filepath): return str(Path(root_dir, filepath))

