from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt

from dcp_client.gui.main_window import MainWindow
from dcp_client.utils.utils import create_warning_box

if TYPE_CHECKING:
    from dcp_client.app import Application

        
class WelcomeWindow(QWidget):
    '''Welcome Window Widget object.
    The first window of the application providing a dialog that allows users to select directories. 
    Currently supported image file types that can be selected for segmentation are: .jpg, .jpeg, .png, .tiff, .tif.
    By clicking 'start' the MainWindow is called.
    '''

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.resize(200, 200)
        self.title = "Select Dataset"
        self.main_layout = QVBoxLayout()
        input_layout = QHBoxLayout()
        label = QLabel(self)
        label.setText('Welcome to Helmholtz AI data centric tool! Please select your dataset folder')
        self.main_layout.addWidget(label)

        self.text_layout = QVBoxLayout()
        self.path_layout = QVBoxLayout()
        self.button_layout = QVBoxLayout()

        val_label = QLabel(self)
        val_label.setText('Uncurated dataset path:')
        inprogr_label = QLabel(self)
        inprogr_label.setText('Curation in progress path:')
        train_label = QLabel(self)
        train_label.setText('Curated dataset path:')
        self.text_layout.addWidget(val_label)
        self.text_layout.addWidget(inprogr_label)
        self.text_layout.addWidget(train_label)

        self.val_textbox = QLineEdit(self)
        self.inprogr_textbox = QLineEdit(self)
        self.train_textbox = QLineEdit(self)
        self.path_layout.addWidget(self.val_textbox)
        self.path_layout.addWidget(self.inprogr_textbox)
        self.path_layout.addWidget(self.train_textbox)
        
        self.file_open_button_val = QPushButton('Browse',self)
        self.file_open_button_val.show()
        self.file_open_button_val.clicked.connect(self.browse_eval_clicked)
        self.file_open_button_prog = QPushButton('Browse',self)
        self.file_open_button_prog.show()
        self.file_open_button_prog.clicked.connect(self.browse_inprogr_clicked)
        self.file_open_button_train = QPushButton('Browse',self)
        self.file_open_button_train.show()
        self.file_open_button_train.clicked.connect(self.browse_train_clicked)
        self.button_layout.addWidget(self.file_open_button_val)
        self.button_layout.addWidget(self.file_open_button_prog)
        self.button_layout.addWidget(self.file_open_button_train)

        input_layout.addLayout(self.text_layout)
        input_layout.addLayout(self.path_layout)
        input_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(input_layout)

        self.start_button = QPushButton('Start', self)
        self.start_button.setFixedSize(120, 30)
        self.start_button.show()
        # check if we need to upload data to server
        self.done_upload = False # we only do once
        if self.app.syncer.host_name == "local":
            self.start_button.clicked.connect(self.start_main)
        else:
            self.start_button.clicked.connect(self.start_upload_and_main)
        self.main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        self.setLayout(self.main_layout)

        self.show()

    def browse_eval_clicked(self):
        '''
        Activates  when the user clicks the button to choose the evaluation directory (QFileDialog) and 
        displays the name of the evaluation directory chosen in the validation textbox line (QLineEdit).
        '''
        self.fd = QFileDialog()
        try:
            self.fd.setFileMode(QFileDialog.Directory)
            if self.fd.exec_():
                self.app.eval_data_path = self.fd.selectedFiles()[0]
            self.val_textbox.setText(self.app.eval_data_path)
        finally:
            self.fd = None
            
    def browse_train_clicked(self):
        '''
        Activates  when the user clicks the button to choose the train directory (QFileDialog) and 
        displays the name of the train directory chosen in the train textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.app.train_data_path = fd.selectedFiles()[0]
        self.train_textbox.setText(self.app.train_data_path)


    def browse_inprogr_clicked(self):
        '''
        Activates  when the user clicks the button to choose the curation in progress directory (QFileDialog) and 
        displays the name of the evaluation directory chosen in the validation textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_(): # Browse clicked
            self.app.inprogr_data_path = fd.selectedFiles()[0] #TODO: case when browse is clicked but nothing is specified - currently it is filled with os.getcwd()
        self.inprogr_textbox.setText(self.app.inprogr_data_path)
  
    def start_main(self):
        '''
        Starts the main window after the user clicks 'Start' and only if both evaluation and train directories are chosen. 
        '''
        
        if self.app.train_data_path and self.app.eval_data_path:
            self.hide()
            self.mw = MainWindow(self.app)
        else:
            message_text = "You need to specify a folder both for your uncurated and curated dataset (even if the curated folder is currently empty). Please go back and select folders for both."
            _ = create_warning_box(message_text, message_title="Warning")

    def start_upload_and_main(self):
        '''
        If the configs are set to use remote not local server then the user is asked to confirm the upload of their data
        to the server and the upload starts before launching the main window.
        '''
        if self.done_upload is False:
            message_text = ("Your current configurations are set to run some operations on the cloud. \n"
                            "For this we need to upload your data to our server."
                            "We will now upload your data. Click ok to continue. \n"
                            "If you do not agree close the application and contact your software provider.")
            usr_response = create_warning_box(message_text, message_title="Warning", add_cancel_btn=True)
            if usr_response: self.app.upload_data_to_server()
            self.done_upload = True
        self.start_main()
    