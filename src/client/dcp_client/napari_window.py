from __future__ import annotations
from typing import List, TYPE_CHECKING

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout
import napari

if TYPE_CHECKING:
    from app import Application

import utils

class NapariWindow(QWidget):
    '''Napari Window Widget object.
    Opens the napari image viewer to view and fix the labeles.
    :param app:
    :type Application
    '''

    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.setWindowTitle("napari viewer")

        # Load image and get corresponding segmentation filenames
        img = self.app.load_image()
        seg_filepaths = self.app.search_segs()

        # Set the viewer
        self.viewer = napari.Viewer(show=False)
        self.viewer.add_image(img, name=utils.get_path_stem(self.app.cur_selected_img))
        for seg_file in seg_filepaths:
            self.viewer.add_labels(self.app.load_image(seg_file), name=utils.get_path_stem(seg_file))

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

        self.setLayout(layout)
        self.show()

    def on_add_to_curated_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        current_dir = utils.get_path_parent(self.app.cur_selected_img)
        if  current_dir == str(self.app.train_data_path):
            message_text = "Image is already in the \'Curated data\' folder and should not be changed again"
            utils.create_warning_box(message_text, message_title="Warning")
            return
        
        # take the name of the currently selected layer (by the user)
        cur_seg_selected = self.viewer.layers.selection.active.name
        # TODO if more than one item is selected this will break!
        if '_seg' not in cur_seg_selected:
            message_text = "Please select the segmenation you wish to save from the layer list"
            utils.create_warning_box(message_text, message_title="Warning")
            return
        seg = self.viewer.layers[cur_seg_selected].data

        # Move original image
        self.app.move_image(current_dir, self.app.train_data_path, utils.get_path_name(self.app.cur_selected_img))

        # Save the (changed) seg
        self.app.save_image(self.app.train_data_path, cur_seg_selected+'.tiff', seg)

        # We remove seg from the current directory if it exists (both eval and inprogr allowed)
        self.app.delete_image(current_dir, cur_seg_selected+'.tiff')
        # TODO Create the Archive folder for the rest? Or move them as well? 

        self.close()

    def on_add_to_inprogress_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        # TODO: Do we allow this? What if they moved it by mistake? User can always manually move from their folders?)
        current_dir = utils.get_path_parent(self.app.cur_selected_img)
        if current_dir == str(self.train_data_path):
            message_text = "Images from '\Curated data'\ folder can not be moved back to \'Curatation in progress\' folder."
            utils.create_warning_box(message_text, message_title="Warning")
            return
        
        # Move original image
        self.app.move_image(current_dir, self.app.inprogr_data_path, utils.get_path_name(self.app.cur_selected_img))
        
        # take the name of the currently selected layer (by the user)
        cur_seg_selected = self.viewer.layers.selection.active.name
        # TODO if more than one item is selected this will break!
        if '_seg' not in cur_seg_selected:
            message_text = "Please select the segmenation you wish to save from the layer list"
            utils.create_warning_box(message_text, message_title="Warning")
            return
        seg = self.viewer.layers[cur_seg_selected].data

        # Save the (changed) seg
        self.app.save_image(self.app.inprogr_data_path, cur_seg_selected+'.tiff', seg)

        # We remove seg from the eval data directory if it exists (both eval and inprogr allowed)
        self.app.delete_image(self.app.eval_data_path, cur_seg_selected+'.tiff')
        # TODO Create the Archive folder for the rest? Or move them as well? 

        self.close()