from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import QTimer

class MyWidget(QWidget):

    msg = None
    sim = False # will be used for testing to simulate user click

    def create_warning_box(self, message_text: str=" ", message_title: str="Information", add_cancel_btn: bool=False, custom_dialog=None) -> None:    
        #setup box
        if custom_dialog is not None: self.msg = custom_dialog
        else: self.msg = QMessageBox()

        if message_title=="Warning": 
            message_type = QMessageBox.Warning
        elif message_title=="Error":
            message_type = QMessageBox.Critical
        else:
            message_type = QMessageBox.Information
        self.msg.setIcon(message_type)
        self.msg.setText(message_text)
        self.msg.setWindowTitle(message_title)
        # if specified add a cancel button else only an ok
        if add_cancel_btn:
            self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            # simulate button click if specified - workaround used for testing
            if self.sim: QTimer.singleShot(0, self.msg.button(QMessageBox.Cancel).clicked)
        else:
            self.msg.setStandardButtons(QMessageBox.Ok)
            # simulate button click if specified - workaround used for testing
            if self.sim: QTimer.singleShot(0, self.msg.button(QMessageBox.Ok).clicked)
        # return if user clicks Ok and False otherwise
        usr_response = self.msg.exec()
        if usr_response == QMessageBox.Ok: return True
        else: return False