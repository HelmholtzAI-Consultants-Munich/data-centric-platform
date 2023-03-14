import os
from pathlib import Path
from typing import List

from skimage.io import imread, imsave
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import napari

class Application():
    '''Contains functions with main code'''

    def __init__(self, 
                img_filename,
                eval_data_path,
                train_data_path,
                inprogr_data_path):
        self.img_filename = img_filename
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path

    def load_image_seg(self):
        self.potential_seg_name = Path(self.img_filename).stem + '_seg.tiff' #+Path(self.img_filename).suffix
        if os.path.exists(os.path.join(self.eval_data_path, self.img_filename)):
            img = imread(os.path.join(self.eval_data_path, self.img_filename))
            if os.path.exists(os.path.join(self.eval_data_path, self.potential_seg_name)):
                seg = imread(os.path.join(self.eval_data_path, self.potential_seg_name))
            else: seg = None
        else: 
            img = imread(os.path.join(self.train_data_path, self.img_filename))
            if os.path.exists(os.path.join(self.train_data_path, self.potential_seg_name)):
                seg = imread(os.path.join(self.train_data_path, self.potential_seg_name))
            else: seg = None
        return img, seg
    
    def save_seg(self, seg, from_directory, to_directory):
        
        os.replace(os.path.join(from_directory, self.img_filename), os.path.join(to_directory, self.img_filename))
        seg_name = Path(self.img_filename).stem+ '_seg.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(to_directory, seg_name), seg)
        if os.path.exists(os.path.join(from_directory, seg_name)): 
            os.remove(os.path.join(from_directory, seg_name))


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
                train_data_path,
                inprogr_data_path):
        super().__init__()
        self.app = Application(
            img_filename = img_filename,
            eval_data_path = eval_data_path,
            train_data_path = train_data_path,
            inprogr_data_path = inprogr_data_path
        )

        img, seg = self.app.load_image_seg()
 
        self.setWindowTitle("napari viewer")
        self.viewer = napari.Viewer(show=False)
        self.viewer.add_image(img)

        if seg is not None: 
            self.viewer.add_labels(seg)

        main_window = self.viewer.window._qt_window
        layout = QVBoxLayout()
        layout.addWidget(main_window)

        buttons_layout = QHBoxLayout()

        add_to_inprogress_button = QPushButton('Move to \'Curatation in progress\' folder')
        buttons_layout.addWidget(add_to_inprogress_button)
        add_to_inprogress_button.clicked.connect(self.on_add_to_inprogress_button_clicked)
    
        add_to_curated_button = QPushButton('Move to \'Curated dataset\' folder')
        buttons_layout.addWidget(add_to_curated_button)
        add_to_curated_button.clicked.connect(self.on_add_to_curated_button_clicked)

        layout.addLayout(buttons_layout)

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


    def on_add_to_curated_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''

        label_names = self._get_layer_names()
        seg = self.viewer.layers[label_names[0]].data
        self.app.save_seg(seg, from_directory=self.app.eval_data_path, to_directory=self.app.train_data_path)
        self.close()

    def on_add_to_inprogress_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''

        label_names = self._get_layer_names()
        seg = self.viewer.layers[label_names[0]].data
        self.app.save_seg(seg, from_directory=self.app.eval_data_path, to_directory=self.app.inprogr_data_path)
        self.close()

    '''
    def on_return_button_clicked(self):
        self.close()
    '''