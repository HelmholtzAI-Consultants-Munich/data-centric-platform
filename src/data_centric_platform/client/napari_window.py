import os
from pathlib import Path
from typing import List

from skimage.io import imread, imsave
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import napari

from utils import create_warning_box

class NapariWindow(QWidget):
    '''Napari Window Widget object.
    Opens the napari image viewer to view and fix the labeles.
    :param img_filepath:
    :type img_filepath: string
    :param eval_data_path:
    :type eval_data_path:
    :param train_data_path:
    :type train_data_path:
    '''

    def __init__(self, 
                img_filepath,
                eval_data_path,
                train_data_path,
                inprogr_data_path):
        super().__init__()
        self.img_filepath = img_filepath
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path

        # Check the directory the image was selected from:
        self.img_directory = Path(self.img_filepath).parent

        # Take all segmentations of the image from the current directory:
        search_string = Path(self.img_filepath).stem + '_seg'
        image_seg_files = []
        files = os.listdir(self.img_directory)
        for file_name in files:
            if search_string in file_name:
                image_seg_files.append(file_name)

        # Read the selected image and read the segmentation if any:
        self.img = imread(self.img_filepath)

        # Set the viewer
        self.setWindowTitle("napari viewer")
        self.viewer = napari.Viewer(show=False)
        self.viewer.add_image(self.img, name=Path(self.img_filepath).stem)

        if image_seg_files:
            for seg_file in image_seg_files: 
                seg = imread(os.path.join(self.img_directory, seg_file))
                self.viewer.add_labels(seg, name=Path(seg_file).stem)

        self.layer_names_at_start = self._get_layer_names()
        print(f"THESE ARE THE LABELS {self.layer_names_at_start}")

        # self.potential_seg_name = Path(self.img_filepath).stem + '_seg.tiff' 
        # if os.path.exists(os.path.join(self.eval_data_path, self.img_filepath)):
        #     self.img = imread(os.path.join(self.eval_data_path, self.img_filepath))
        #     if os.path.exists(os.path.join(self.eval_data_path, self.potential_seg_name)):
        #         seg = imread(os.path.join(self.eval_data_path, self.potential_seg_name))
        #     else: seg = None
        # else: 
        #     self.img = imread(os.path.join(self.train_data_path, self.img_filepath))
        #     if os.path.exists(os.path.join(self.train_data_path, self.potential_seg_name)):
        #         seg = imread(os.path.join(self.train_data_path, self.potential_seg_name))
        #     else: seg = None
        
        # self.setWindowTitle("napari viewer")
        # self.viewer = napari.Viewer(show=False)
        # self.viewer.add_image(self.img)

        # if seg is not None: 
        #     self.viewer.add_labels(seg)

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
        
    def _compare_layer_names(self, layer_names_end) -> List[int]:
        
        return [i for i in range(len(self.layer_names_at_start)) if self.layer_names_at_start[i] != layer_names_end[i]]


    def on_add_to_curated_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''

        if str(self.img_directory) == str(self.train_data_path):
            message_text = "Image is already in the \'Curated data\' folder and should not be changed again"
            create_warning_box(message_text, message_title="Warning")
            return

        layer_names_end = self._get_layer_names()
        print(f"THESE ARE THE LABELS AT THE END{layer_names_end}")
        # Compare the change in layer names:
        changed_segs = self._compare_layer_names(layer_names_end)
        print(f"THESE ARE CHANGED SEGS INDICES {changed_segs}")
        
        if changed_segs: # Take changed
            seg_name = layer_names_end[changed_segs[0]] + '.tiff'
            seg = self.viewer.layers[layer_names_end[changed_segs[0]]].data
        else: # Take the first
            seg_name = layer_names_end[0] + '.tiff'
            seg = self.viewer.layers[layer_names_end[0]].data
        
        print(f"THIS IS THE SEG IT IS TAKING {seg_name}")
        # Move original image
        os.replace(self.img_filepath, os.path.join(self.train_data_path, Path(self.img_filepath).name))

        # Save the (changed) seg
        imsave(os.path.join(self.train_data_path, seg_name),seg)

        # We take from img_directory (both eval and inprogr allowed)
        if os.path.exists(os.path.join(self.img_directory, seg_name)): 
            os.replace(os.path.join(self.img_directory, seg_name), os.path.join(self.train_data_path, seg_name))

        # TODO Create the Archive folder for the rest? Or move them as well? 
        
        self.close()

    def on_add_to_inprogress_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        # TODO: Do we allow this? What if they moved it by mistake? User can always manually move from their folders?)
        if str(self.img_directory) == str(self.train_data_path):
            message_text = "Images from '\Curated data'\ folder can not be moved back to \'Curatation in progress\' folder."
            create_warning_box(message_text, message_title="Warning")
            return

        layer_names_end = self._get_layer_names()
        print(f"THESE ARE THE LABELS AT THE END{layer_names_end}")
        # Compare the change in layer names:
        changed_segs = self._compare_layer_names(layer_names_end)
        print(f"THESE ARE CHANGED SEGS INDICES {changed_segs}")
        
        if changed_segs: # Take changed
            seg_name = layer_names_end[changed_segs[0]] + '.tiff'
            seg = self.viewer.layers[layer_names_end[changed_segs[0]]].data
        else: # Take the first
            seg_name = layer_names_end[0] + '.tiff'
            seg = self.viewer.layers[layer_names_end[0]].data
        
        print(f"THIS IS THE SEG IT IS TAKING {seg_name}")
        # Move original image
        os.replace(self.img_filepath, os.path.join(self.inprogr_data_path, Path(self.img_filepath).name))

        # Save the (changed) seg
        imsave(os.path.join(self.inprogr_data_path, seg_name),seg)

        # We take from eval_data_path
        if os.path.exists(os.path.join(self.eval_data_path, seg_name)): 
            os.replace(os.path.join(self.eval_data_path, seg_name), os.path.join(self.inprogr_data_path, seg_name))

        # TODO Create the Archive folder for the rest? Or move them as well? 

        self.close()

    '''
    def on_return_button_clicked(self):
        self.close()
    '''