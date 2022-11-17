import sys
import os
from pathlib import Path
from typing import List
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMainWindow, QFileSystemModel, QListView, QHBoxLayout, QFileIconProvider, QLabel, QFileDialog, QLineEdit, QTreeView
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
import napari
from skimage.io import imread, imsave
from cellpose import models, utils
import torch 
from skimage.transform import resize
import numpy as np
from skimage.morphology import dilation, disk

from matplotlib import pyplot as plt

ICON_SIZE = QSize(128,128)
accepted_types = (".jpg",".tiff",".png", ".tif")

def changeWindow(w1, w2):
    w1.hide()
    w2.show()

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
    def __init__(self, 
                img_filename,
                eval_data_path,
                train_data_path):
        super().__init__()
        self.img_filename = img_filename
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        potential_seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
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
        
        
        self.setWindowTitle("napari Viewer")
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.img)
        if seg is not None: self.viewer.add_labels(seg)
        main_window = self.viewer.window._qt_window
        layout = QVBoxLayout()
        layout.addWidget(main_window)
        add_button = QPushButton('Add to training data')
        layout.addWidget(add_button)
        #self.return_button = QPushButton('Return')
        #layout.addWidget(self.return_button)

        add_button.clicked.connect(self.on_add_button_clicked)
        #self.return_button.clicked.connect(self.on_return_button_clicked)
        self.setLayout(layout)
        self.show()

    def _get_layer_names(self, layer_type: napari.layers.Layer = napari.layers.Labels) -> List[str]:
        """
        Get list of layer names of a given layer type.
        """
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
        label_names = self._get_layer_names()
        seg = self.viewer.layers[label_names[0]].data
        os.replace(os.path.join(self.eval_data_path, self.img_filename), os.path.join(self.train_data_path, self.img_filename))
        seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
        imsave(os.path.join(self.train_data_path, seg_name),seg)
        if os.path.exists(os.path.join(self.eval_data_path, seg_name)): 
            os.remove(os.path.join(self.eval_data_path, seg_name))
        self.close()

    '''
    def on_return_button_clicked(self):
        self.close()
    '''

class MainWindow(QWidget):
    def __init__(self, eval_data_path, train_data_path):
        super().__init__()

        self.title = "Data Overview"
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.main_window()

    def main_window(self):
        self.setWindowTitle(self.title)
        #self.resize(1000, 1500)
        self.main_layout = QVBoxLayout()  
        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()

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
        self.top_layout.addLayout(self.eval_dir_layout)

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
        self.top_layout.addLayout(self.train_dir_layout)

        self.main_layout.addLayout(self.top_layout)
        
        # add buttons
        self.launch_nap_button = QPushButton("View image and fix label", self)
        self.launch_nap_button.clicked.connect(self.launch_napari_window)  # add selected image    
        self.bottom_layout.addWidget(self.launch_nap_button)
        
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)  # add selected image    
        self.bottom_layout.addWidget(self.train_button)

        self.inference_button = QPushButton("Generate Labels", self)
        self.inference_button.clicked.connect(self.run_inference)  # add selected image    
        self.bottom_layout.addWidget(self.inference_button)

        self.main_layout.addLayout(self.bottom_layout)

        self.setLayout(self.main_layout)
        self.show()

    def launch_napari_window(self):                                       
        self.nap_win = NapariWindow(img_filename=self.cur_selected_img, 
                                    eval_data_path=self.eval_data_path, 
                                    train_data_path=self.train_data_path)
        self.nap_win.show()

    def item_eval_selected(self, item):
        self.cur_selected_img = item.data()
    
    def item_train_selected(self, item):
        self.cur_selected_img = item.data()

    def train_model(self):
        pass

    def run_inference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device=="cuda":
            model = models.Cellpose(gpu=True, model_type="cyto")
        else:
            model = models.Cellpose(gpu=False, model_type="cyto")
        list_files = [file for file in os.listdir(self.eval_data_path) if file.endswith('.tif')]
        for img_filename in list_files:
            if '_seg' in img_filename:  
                continue
                #extend to check the prefix also matches an existing image
                #seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
            else:
                img = imread(os.path.join(self.eval_data_path, img_filename))
                orig_size = img.shape
                img = resize(img, (384, 512))
                mask, _, _, _ = model.eval(img)
                seg_name = Path(img_filename).stem+'_seg'+Path(img_filename).suffix
                mask = resize(mask, orig_size, order=0)
                outlines = utils.masks_to_outlines(mask) #[True, False] outputs
                #outlines = dilation(outlines, disk(4))
                new_mask = mask.copy()
                
                #mask[mask!=0] = 125
                imsave('temp.tif', np.logical_and(mask, outlines))
                new_mask[mask!=0] = 2
                new_mask[outlines==True] = 1
                imsave(os.path.join(self.eval_data_path, seg_name), new_mask)
                #imsave(os.path.join(self.eval_data_path, seg_name), mask)


class WelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(200, 200)
        self.title = "Select Dataset"
        self.main_layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setText('Welcome to Helmholtz AI data centric tool! Please select your dataset folder')
        
        self.val_layout = QHBoxLayout()
        self.val_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_eval_clicked)
        self.val_layout.addWidget(self.val_textbox)
        self.val_layout.addWidget(self.fileOpenButton)

        self.train_layout = QHBoxLayout()
        self.train_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_train_clicked)
        self.train_layout.addWidget(self.train_textbox)
        self.train_layout.addWidget(self.fileOpenButton)

        self.main_layout.addWidget(self.label)
        self.main_layout.addLayout(self.val_layout)
        self.main_layout.addLayout(self.train_layout)

        self.start_button = QPushButton('Start', self)
        self.start_button.show()
        self.start_button.clicked.connect(self.start_main)
        self.main_layout.addWidget(self.start_button)
        self.setLayout(self.main_layout)
        self.show()

    def browse_eval_clicked(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.filename_val = fd.selectedFiles()[0]
        self.val_textbox.setText(self.filename_val)
    
    def browse_train_clicked(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.filename_train = fd.selectedFiles()[0]
        self.train_textbox.setText(self.filename_train)

    
    def start_main(self):
        self.hide()
        self.mw = MainWindow(self.filename_val, self.filename_train)
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    sys.exit(app.exec())
