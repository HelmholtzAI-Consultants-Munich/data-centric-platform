import sys
import os
from pathlib import Path
from typing import List
import asyncio

from skimage.io import imread, imsave
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileSystemModel, QHBoxLayout, QFileIconProvider, QLabel, QFileDialog, QLineEdit, QTreeView, QMessageBox
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QPixmap, QIcon
import napari
from bentoml.client import Client
#from server.local_inference import run_inference
import warnings
warnings.simplefilter('ignore')

ICON_SIZE = QSize(512,512)
accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")

client = Client.from_url("http://0.0.0.0:7010") # have the url of the bentoml service here

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()

    def icon(self, type: 'QFileIconProvider.IconType'):

        fn = type.filePath()

        if fn.endswith(accepted_types):
            a = QPixmap(ICON_SIZE)
            a.load(fn)
            return QIcon(a)
        else:
            return super().icon(type)


class NapariWindow(QWidget):
    '''Napari Window Widget object.
    Opens the napari image viewer to view and fix the labeles.
    :param img_filename:
    :type img_filename: string
    :param eval_data_path:
    :type eval_data_path:
    :param train_data_path:
    :type train_data_path:
    '''

    def __init__(self, 
                img_filename,
                eval_data_path,
                train_data_path):
        super().__init__()
        self.img_filename = img_filename
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path

 
        potential_seg_name = Path(self.img_filename).stem + '_seg.tiff' #+Path(self.img_filename).suffix
        if os.path.exists(os.path.join(self.eval_data_path, self.img_filename)):
            self.img = imread(os.path.join(self.eval_data_path, self.img_filename))
            if os.path.exists(os.path.join(self.eval_data_path, potential_seg_name)):
                seg = imread(os.path.join(self.eval_data_path, potential_seg_name))
            else: seg = None
        else: 
            self.img = imread(os.path.join(self.train_data_path, self.img_filename))
            if os.path.exists(os.path.join(self.train_data_path, potential_seg_name)):
                seg = imread(os.path.join(self.train_data_path, potential_seg_name))
            else: seg = None
        
        self.setWindowTitle("napari viewer")
        self.viewer = napari.Viewer(show=False)
        self.viewer.add_image(self.img)

        if seg is not None: 
            self.viewer.add_labels(seg)

        main_window = self.viewer.window._qt_window
        layout = QVBoxLayout()
        layout.addWidget(main_window)

        add_button = QPushButton('Add to training data')
        layout.addWidget(add_button)
        add_button.clicked.connect(self.on_add_button_clicked)

        #self.return_button = QPushButton('Return')
        #layout.addWidget(self.return_button)
        #self.return_button.clicked.connect(self.on_return_button_clicked)

        self.setLayout(layout)
        self.show()


    def _get_layer_names(self, layer_type: napari.layers.Layer = napari.layers.Labels) -> List[str]:
        '''
        Get list of layer names of a given layer type.
        '''
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if type(layer) == layer_type
        ]
        if layer_names:
            return [] + layer_names
        else:
            return []


    def on_add_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''

        label_names = self._get_layer_names()
        seg = self.viewer.layers[label_names[0]].data
        os.replace(os.path.join(self.eval_data_path, self.img_filename), os.path.join(self.train_data_path, self.img_filename))
        seg_name = Path(self.img_filename).stem+ '_seg.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(self.train_data_path, seg_name),seg)
        if os.path.exists(os.path.join(self.eval_data_path, seg_name)): 
            os.remove(os.path.join(self.eval_data_path, seg_name))
        self.close()

    '''
    def on_return_button_clicked(self):
        self.close()
    '''


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

        self.title = "Data Overview"
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path
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
        self.list_view_eval.setRootIndex(model_eval.setRootPath(self.eval_data_path)) 
        self.list_view_eval.clicked.connect(self.item_eval_selected)
        self.cur_selected_img = None
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
        self.list_view_inprogr.setRootIndex(model_inprogr.setRootPath(self.inprogr_data_path)) 
        self.list_view_inprogr.clicked.connect(self.item_inprogr_selected)
        self.inprogr_dir_layout.addWidget(self.list_view_inprogr)
        self.inprogress_layout.addLayout(self.inprogr_dir_layout)

        self.launch_nap_button = QPushButton("View image and fix label", self)
        self.launch_nap_button.clicked.connect(self.launch_napari_window)  # add selected image    
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
        self.list_view_train.setRootIndex(model_train.setRootPath(self.train_data_path)) 
        self.list_view_train.clicked.connect(self.item_train_selected)
        self.train_dir_layout.addWidget(self.list_view_train)
        self.curated_layout.addLayout(self.train_dir_layout)
        
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.on_train_button_clicked)  # add selected image    
        self.curated_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)

        self.main_layout.addLayout(self.curated_layout)

        self.setLayout(self.main_layout)
        self.show()

    def launch_napari_window(self):   
        ''' 
        Launches the napari window after the image is selected.
        '''

        if not self.cur_selected_img or '_seg.tiff' in self.cur_selected_img:
            message_text = "Please first select an image you wish to visualise. The selected image must be an original images, not a mask."
            create_warning_box(message_text, message_title="Warning")
        else:
            self.nap_win = NapariWindow(img_filename=self.cur_selected_img, 
                                        eval_data_path=self.eval_data_path, 
                                        train_data_path=self.train_data_path)
            self.nap_win.show()

    def item_eval_selected(self, item):
        self.cur_selected_img = item.data()
    
    def item_train_selected(self, item):
        self.cur_selected_img = item.data()

    def item_inprogr_selected(self, item):
        self.cur_selected_img = item.data()

    async def _run_train(self):
        response = await client.async_retrain(self.train_data_path)
        return response

    def on_train_button_clicked(self):
        message_text = asyncio.run(self._run_train())
        create_warning_box(message_text)

    async def _run_inference(self):
        response = await client.async_segment_image(self.eval_data_path)
        return response

    def on_run_inference_button_clicked(self):
        list_of_files_not_suported = asyncio.run(self._run_inference())
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            message_text = "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: ', accepted_types. The files that were not supported are: " + ", ".join(list_of_files_not_suported)
            create_warning_box(message_text)
        else:
            create_warning_box("Success! Masks generated for all images", message_title="Success")

def create_warning_box(message_text, message_title="Warning"):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(message_text)
    msg.setWindowTitle(message_title)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()

class WelcomeWindow(QWidget):
    '''Welcome Window Widget object.
    The first window of the application providing a dialog that allows users to select directories. 
    Currently supported image file types that can be selected for segmentation are: .jpg, .jpeg, .png, .tiff, .tif.
    By clicking 'start' the MainWindow is called.
    '''

    def __init__(self):
        super().__init__()
        self.resize(200, 200)
        self.title = "Select Dataset"
        self.main_layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setText('Welcome to Helmholtz AI data centric tool! Please select your dataset folder')
        
        self.val_layout = QHBoxLayout()
        val_label = QLabel(self)
        val_label.setText('Uncurated dataset path:')
        self.val_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_eval_clicked)
        self.val_layout.addWidget(val_label)
        self.val_layout.addWidget(self.val_textbox)
        self.val_layout.addWidget(self.fileOpenButton)

        self.inprogr_layout = QHBoxLayout()
        inprogr_label = QLabel(self)
        inprogr_label.setText('Curation in progress path:')
        self.inprogr_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_inprogr_clicked)
        self.inprogr_layout.addWidget(inprogr_label)
        self.inprogr_layout.addWidget(self.inprogr_textbox)
        self.inprogr_layout.addWidget(self.fileOpenButton)

        self.train_layout = QHBoxLayout()
        train_label = QLabel(self)
        train_label.setText('Curated dataset path:')
        self.train_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_train_clicked)
        self.train_layout.addWidget(train_label)
        self.train_layout.addWidget(self.train_textbox)
        self.train_layout.addWidget(self.fileOpenButton)

        self.main_layout.addWidget(self.label)
        self.main_layout.addLayout(self.val_layout)
        self.main_layout.addLayout(self.inprogr_layout)
        self.main_layout.addLayout(self.train_layout)

        self.start_button = QPushButton('Start', self)
        self.start_button.setFixedSize(120, 30)
        self.start_button.show()
        self.start_button.clicked.connect(self.start_main)
        self.main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        self.setLayout(self.main_layout)

        self.filename_train = ''
        self.filename_val = ''
        #self.filename_inprogr = os.getcwd() #TODO: what is the inprogress path if nothing is specified?
        self.filename_inprogr = ''

        self.show()

    def browse_eval_clicked(self):
        '''
        Activates  when the user clicks the button to choose the evaluation directory (QFileDialog) and 
        displays the name of the evaluation directory chosen in the validation textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.filename_val = fd.selectedFiles()[0]
        self.val_textbox.setText(self.filename_val)
    
    def browse_train_clicked(self):
        '''
        Activates  when the user clicks the button to choose the train directory (QFileDialog) and 
        displays the name of the train directory chosen in the train textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.filename_train = fd.selectedFiles()[0]
        self.train_textbox.setText(self.filename_train)


    def browse_inprogr_clicked(self):
        '''
        Activates  when the user clicks the button to choose the curation in progress directory (QFileDialog) and 
        displays the name of the evaluation directory chosen in the validation textbox line (QLineEdit).
        '''

        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_(): # Browse clicked
            self.filename_inprogr = fd.selectedFiles()[0] #TODO: case when browse is clicked but nothing is specified - currently it is filled with os.getcwd()
        self.inprogr_textbox.setText(self.filename_inprogr)
  
    
    def start_main(self):
        '''
        Starts the main window after the user clicks 'Start' and only if both evaluation and train directories are chosen. 
        '''
        
        if self.filename_train and self.filename_val:
            self.hide()
            self.mw = MainWindow(self.filename_val, self.filename_train, self.filename_inprogr)
        else:
            message_text = "You need to specify a folder both for your uncurated and curated dataset (even if the curated folder is currently empty). Please go back and select folders for both."
            create_warning_box(message_text, message_title="Warning")
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    sys.exit(app.exec())