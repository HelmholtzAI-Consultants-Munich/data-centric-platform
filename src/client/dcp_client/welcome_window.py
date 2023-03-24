from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt

from app import MainWindowApplication
from main_window import MainWindow
from utils import create_warning_box

if TYPE_CHECKING:
    from app import WelcomeApplication

        
class WelcomeWindow(QWidget):
    '''Welcome Window Widget object.
    The first window of the application providing a dialog that allows users to select directories. 
    Currently supported image file types that can be selected for segmentation are: .jpg, .jpeg, .png, .tiff, .tif.
    By clicking 'start' the MainWindow is called.
    '''

    def __init__(self, app: WelcomeApplication):
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
        
        file_open_button_val = QPushButton('Browse',self)
        file_open_button_val.show()
        file_open_button_val.clicked.connect(self.browse_eval_clicked)
        file_open_button_prog = QPushButton('Browse',self)
        file_open_button_prog.show()
        file_open_button_prog.clicked.connect(self.browse_inprogr_clicked)
        file_open_button_train = QPushButton('Browse',self)
        file_open_button_train.show()
        file_open_button_train.clicked.connect(self.browse_train_clicked)
        self.button_layout.addWidget(file_open_button_val)
        self.button_layout.addWidget(file_open_button_prog)
        self.button_layout.addWidget(file_open_button_train)

        input_layout.addLayout(self.text_layout)
        input_layout.addLayout(self.path_layout)
        input_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(input_layout)

        self.start_button = QPushButton('Start', self)
        self.start_button.setFixedSize(120, 30)
        self.start_button.show()
        self.start_button.clicked.connect(self.start_main)
        self.main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        self.setLayout(self.main_layout)

        # self.filename_train = ''
        # self.filename_val = ''
        # #self.app.filename_inprogr = os.getcwd() #TODO: what is the inprogress path if nothing is specified?
        # self.filename_inprogr = ''

        self.show()

    def browse_eval_clicked(self):
        '''
        Activates  when the user clicks the button to choose the evaluation directory (QFileDialog) and 
        displays the name of the evaluation directory chosen in the validation textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.app.filename_val = fd.selectedFiles()[0]
        self.val_textbox.setText(self.app.filename_val)
    
    def browse_train_clicked(self):
        '''
        Activates  when the user clicks the button to choose the train directory (QFileDialog) and 
        displays the name of the train directory chosen in the train textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.app.filename_train = fd.selectedFiles()[0]
        self.train_textbox.setText(self.app.filename_train)


    def browse_inprogr_clicked(self):
        '''
        Activates  when the user clicks the button to choose the curation in progress directory (QFileDialog) and 
        displays the name of the evaluation directory chosen in the validation textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_(): # Browse clicked
            self.app.filename_inprogr = fd.selectedFiles()[0] #TODO: case when browse is clicked but nothing is specified - currently it is filled with os.getcwd()
        self.inprogr_textbox.setText(self.app.filename_inprogr)
  
    
    def start_main(self):
        '''
        Starts the main window after the user clicks 'Start' and only if both evaluation and train directories are chosen. 
        '''
        
        if self.app.filename_train and self.app.filename_val:
            self.hide()
            app = MainWindowApplication('', '', '')
            self.mw = MainWindow(self.app.filename_val, self.app.filename_train, self.app.filename_inprogr)
        else:
            message_text = "You need to specify a folder both for your uncurated and curated dataset (even if the curated folder is currently empty). Please go back and select folders for both."
            create_warning_box(message_text, message_title="Warning")
    