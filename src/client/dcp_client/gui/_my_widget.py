from PyQt5.QtWidgets import QWidget, QMessageBox
from PyQt5.QtCore import QTimer

class MyWidget(QWidget):
    """
    This class represents a custom widget.
    """

    msg = None
    sim = False  # will be used for testing to simulate user click

    def create_warning_box(
        self,
        message_text: str = " ",
        message_title: str = "Information",
        add_cancel_btn: bool = False,
        custom_dialog=None,
    ) -> None:
        """Creates a warning box with the specified message and options.

        :param message_text: The text to be displayed in the message box.
        :type message_text: str
        :param message_title: The title of the message box. Default is "Information".
        :type message_title: str
        :param add_cancel_btn: Flag indicating whether to add a cancel button to the message box. Default is False.
        :type add_cancel_btn: bool
        :param custom_dialog: An optional custom dialog to use instead of creating a new QMessageBox instance. Default is None.
        :type custom_dialog: Any
        :return: None
        """
        # setup box
        if custom_dialog is not None:
            self.msg = custom_dialog
        else:
            self.msg = QMessageBox()

        if message_title == "Warning":
            message_type = QMessageBox.Warning
        elif message_title == "Error":
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
            if self.sim:
                QTimer.singleShot(0, self.msg.button(QMessageBox.Cancel).clicked)
        else:
            self.msg.setStandardButtons(QMessageBox.Ok)
            # simulate button click if specified - workaround used for testing
            if self.sim:
                QTimer.singleShot(0, self.msg.button(QMessageBox.Ok).clicked)
        # return if user clicks Ok and False otherwise
        usr_response = self.msg.exec()
        if usr_response == QMessageBox.Ok:
            return True
        else:
            return False

    def create_selection_box(
        self,
        message_text: str = " ",
        message_title: str = "Selection",
        custom_dialog=None,
    ) -> None:
        """Creates a selection box with Ok and Cancel buttons.

        - Clicking Ok returns a message that an action needs to take place.
        - Clicking Cancel closes the box and does nothing.

        :param message_text: The text to be displayed in the message box.
        :type message_text: str
        :param message_title: The title of the message box. Default is "Selection".
        :type message_title: str
        :param custom_dialog: An optional custom dialog to use instead of creating a new QMessageBox instance.
        :type custom_dialog: Any
        :return: None
        """
        # setup box
        if custom_dialog is not None:
            self.msg = custom_dialog
        else:
            self.msg = QMessageBox()

        self.msg.setIcon(QMessageBox.Question)
        self.msg.setText(message_text)
        self.msg.setWindowTitle(message_title)

        # always Ok and Cancel buttons
        self.msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)

        # simulate button click if testing
        if hasattr(self, "sim") and self.sim:
            QTimer.singleShot(0, self.msg.button(QMessageBox.Ok).clicked)

         # execute the dialog
        usr_response = self.msg.exec()

        if usr_response == QMessageBox.Ok: return "action"
        # Cancel does nothing
        else: return "do-nothing"
