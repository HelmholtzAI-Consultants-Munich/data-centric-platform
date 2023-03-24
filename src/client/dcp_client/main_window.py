import asyncio
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QFileSystemModel, QHBoxLayout, QLabel, QTreeView
from PyQt5.QtCore import Qt
from bentoml.client import Client

import settings
from utils import IconProvider, create_warning_box
from napari_window import NapariWindow
from app import Napari_Application






class Application:
    def __init__(self, eval_data_path, train_data_path, inprogr_data_path) -> None:
    
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path
        self.client = Client.from_url("http://0.0.0.0:7010") # have the url of the bentoml service here
        self.cur_selected_img = None
    
    # run train
    async def _run_train(self):
        response = await self.client.async_retrain(self.train_data_path)
        return response
    
    def run_train(self):
        return asyncio.run(self._run_train())
    
    async def _run_inference(self):
        response = await self.app.client.async_segment_image(self.app.eval_data_path)
        return response
    
    def run_inference(self):
        list_of_files_not_suported = asyncio.run(self._run_inference())
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            message_text = "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: ", settings.accepted_types, "The files that were not supported are: " + ", ".join(list_of_files_not_suported)
            message_title = "Warning"
        else:
            message_text = "Success! Masks generated for all images"
            message_title="Success"
        return message_text, message_title

class MainWindow(QWidget):
    '''Main Window Widget object.
    Opens the main window of the app where selected images in both directories are listed. 
    User can view the images, train the mdoel to get the labels, and visualise the result.
    :param eval_data_path: Chosen path to images without labeles, selected by the user in the WelcomeWindow
    :type eval_data_path: string
    :param train_data_path: Chosen path to images with labeles, selected by the user in the WelcomeWindow
    :type train_data_path: string
    '''

    def __init__(self, eval_data_path, train_data_path, inprogr_data_path):
        super().__init__()
        self.app = Application(eval_data_path, train_data_path, inprogr_data_path)
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
        self.inference_button.clicked.connect(self.on_inference_button_clicked)  # add selected image    
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
        self.list_view_train.clicked.connect(self.on_train_item_selected)
        self.train_dir_layout.addWidget(self.list_view_train)
        self.curated_layout.addLayout(self.train_dir_layout)
        
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.on_train_button_clicked)  # add selected image    
        self.curated_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.curated_layout)

        self.setLayout(self.main_layout)
        self.show()

    def on_train_item_selected(self, item):
        self.app.cur_selected_img = item.data()

    def on_item_eval_selected(self, item):
        self.app.cur_selected_img = item.data()
    
    def on_item_inprogr_selected(self, item):
        self.app.cur_selected_img = item.data()

    def on_train_button_clicked(self):
        message_text = self.app.run_train()
        create_warning_box(message_text)

    def on_run_inference_button_clicked(self):
        list_of_files_not_suported = self.app.run_inference()
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            message_text = "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: ", settings.accepted_types, "The files that were not supported are: " + ", ".join(list_of_files_not_suported)

    def on_launch_napari_button_clicked(self):   
        ''' 
        Launches the napari window after the image is selected.
        '''
        if not self.app.cur_selected_img or '_seg.tiff' in self.app.cur_selected_img:
            message_text = "Please first select an image you wish to visualise. The selected image must be an original images, not a mask."
            create_warning_box(message_text, message_title="Warning")
        else:
            napari_app = Napari_Application(img_filename=self.cur_selected_img, 
                                        eval_data_path=self.eval_data_path, 
                                        train_data_path=self.train_data_path,
                                        inprogr_data_path=self.inprogr_data_path)
            self.nap_win = NapariWindow(napari_app)
            self.nap_win.show()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = MainWindow('', '','')
    sys.exit(app.exec())