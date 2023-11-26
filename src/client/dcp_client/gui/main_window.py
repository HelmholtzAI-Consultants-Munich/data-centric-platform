from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileSystemModel, QHBoxLayout, QLabel, QTreeView
from PyQt5.QtCore import Qt

from dcp_client.utils import settings
from dcp_client.utils.utils import IconProvider, create_warning_box
from dcp_client.gui.napari_window import NapariWindow

if TYPE_CHECKING:
    from dcp_client.app import Application


class MainWindow(QWidget):
    '''Main Window Widget object.
    Opens the main window of the app where selected images in both directories are listed. 
    User can view the images, train the model to get the labels, and visualise the result.
    :param eval_data_path: Chosen path to images without labeles, selected by the user in the WelcomeWindow
    :type eval_data_path: string
    :param train_data_path: Chosen path to images with labeles, selected by the user in the WelcomeWindow
    :type train_data_path: string
    '''

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.title = "Data Overview"
        self.main_window()
        
    def main_window(self):
        self.setWindowTitle(self.title)
        #self.resize(1000, 1500)
        self.main_layout = QHBoxLayout()  
        
        self.uncurated_layout = QVBoxLayout()
        self.inprogress_layout = QVBoxLayout()
        self.curated_layout = QVBoxLayout()

        self.eval_dir_layout = QVBoxLayout() 
        self.eval_dir_layout.setContentsMargins(0,0,0,0)
        self.label_eval = QLabel(self)
        self.label_eval.setText("Uncurated dataset")
        self.eval_dir_layout.addWidget(self.label_eval)
        # add eval dir list
        model_eval = QFileSystemModel()
        model_eval.setIconProvider(IconProvider())
        self.list_view_eval = QTreeView(self)
        self.list_view_eval.setModel(model_eval)
        for i in range(1,4):
            self.list_view_eval.hideColumn(i)
        #self.list_view_eval.setFixedSize(600, 600)
        self.list_view_eval.setRootIndex(model_eval.setRootPath(self.app.eval_data_path)) 
        self.list_view_eval.clicked.connect(self.on_item_eval_selected)
        
        self.eval_dir_layout.addWidget(self.list_view_eval)
        self.uncurated_layout.addLayout(self.eval_dir_layout)

        # add buttons
        self.inference_button = QPushButton("Generate Labels", self)
        self.inference_button.clicked.connect(self.on_run_inference_button_clicked)  # add selected image    
        self.uncurated_layout.addWidget(self.inference_button, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.uncurated_layout)

        # In progress layout
        self.inprogr_dir_layout = QVBoxLayout() 
        self.inprogr_dir_layout.setContentsMargins(0,0,0,0)
        self.label_inprogr = QLabel(self)
        self.label_inprogr.setText("Curation in progress")
        self.inprogr_dir_layout.addWidget(self.label_inprogr)
        # add in progress dir list
        model_inprogr = QFileSystemModel()
        #self.list_view = QListView(self)
        self.list_view_inprogr = QTreeView(self)
        model_inprogr.setIconProvider(IconProvider())
        self.list_view_inprogr.setModel(model_inprogr)
        for i in range(1,4):
            self.list_view_inprogr.hideColumn(i)
        #self.list_view_inprogr.setFixedSize(600, 600)
        self.list_view_inprogr.setRootIndex(model_inprogr.setRootPath(self.app.inprogr_data_path)) 
        self.list_view_inprogr.clicked.connect(self.on_item_inprogr_selected)
        self.inprogr_dir_layout.addWidget(self.list_view_inprogr)
        self.inprogress_layout.addLayout(self.inprogr_dir_layout)

        self.launch_nap_button = QPushButton("View image and fix label", self)
        self.launch_nap_button.clicked.connect(self.on_launch_napari_button_clicked)  # add selected image    
        self.inprogress_layout.addWidget(self.launch_nap_button, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.inprogress_layout)

        # Curated layout
        self.train_dir_layout = QVBoxLayout() 
        self.train_dir_layout.setContentsMargins(0,0,0,0)
        self.label_train = QLabel(self)
        self.label_train.setText("Curated dataset")
        self.train_dir_layout.addWidget(self.label_train)
        # add train dir list
        model_train = QFileSystemModel()
        #self.list_view = QListView(self)
        self.list_view_train = QTreeView(self)
        model_train.setIconProvider(IconProvider())
        self.list_view_train.setModel(model_train)
        for i in range(1,4):
            self.list_view_train.hideColumn(i)
        #self.list_view_train.setFixedSize(600, 600)
        self.list_view_train.setRootIndex(model_train.setRootPath(self.app.train_data_path)) 
        self.list_view_train.clicked.connect(self.on_item_train_selected)
        self.train_dir_layout.addWidget(self.list_view_train)
        self.curated_layout.addLayout(self.train_dir_layout)
        
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.on_train_button_clicked)  # add selected image    
        self.curated_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.curated_layout)

        self.setLayout(self.main_layout)
        self.show()

    def on_item_train_selected(self, item):
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.train_data_path

    def on_item_eval_selected(self, item):
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.eval_data_path

    def on_item_inprogr_selected(self, item):
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.inprogr_data_path

    def on_train_button_clicked(self):
        message_text = self.app.run_train()
        create_warning_box(message_text)

    def on_run_inference_button_clicked(self):
        message_text, message_title = self.app.run_inference()
        create_warning_box(message_text, message_title)

    def on_launch_napari_button_clicked(self):   
        ''' 
        Launches the napari window after the image is selected.
        '''
        if not self.app.cur_selected_img or '_seg.tiff' in self.app.cur_selected_img:
            message_text = "Please first select an image you wish to visualise. The selected image must be an original images, not a mask."
            create_warning_box(message_text, message_title="Warning")
        else:
            self.nap_win = NapariWindow(self.app)
            self.nap_win.show()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from app import Application
    from bentoml_model import BentomlModel
    from fsimagestorage import FilesystemImageStorage
    import settings
    settings.init()
    image_storage = FilesystemImageStorage()
    ml_model = BentomlModel()
    app = QApplication(sys.argv)
    app_ = Application(ml_model=ml_model, 
                       image_storage=image_storage, 
                       server_ip='0.0.0.0',
                       server_port=7010,
                       eval_data_path='data', 
                       train_data_path='', # set path
                       inprogr_data_path='') # set path
    window = MainWindow(app=app_)
    sys.exit(app.exec())