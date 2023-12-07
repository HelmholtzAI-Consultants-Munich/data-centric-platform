from __future__ import annotations
from typing import List, TYPE_CHECKING

from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QGridLayout
from PyQt5.QtCore import Qt
import napari

if TYPE_CHECKING:
    from dcp_client.app import Application

from dcp_client.utils import utils
from napari.qt import thread_worker
import numpy as np

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
        self.app.search_segs()

        # Set the viewer
        self.viewer = napari.Viewer(show=False)

        self.viewer.add_image(img, name=utils.get_path_stem(self.app.cur_selected_img))

        for seg_file in self.app.seg_filepaths:
            self.viewer.add_labels(self.app.load_image(seg_file), name=utils.get_path_stem(seg_file))

        layer = self.viewer.layers[utils.get_path_stem(self.app.seg_filepaths[0])]

        self.changed = False

        main_window = self.viewer.window._qt_window
        layout = QGridLayout()
        layout.addWidget(main_window, 0, 0, 1, 4)

        # set first mask as active by default
        self.active_mask_index = 0

        if layer.data.shape[0] >= 2:
            # User hint
            message_label = QLabel('Choose an active mask')
            message_label.setAlignment(Qt.AlignRight)
            layout.addWidget(message_label, 1, 0)

        # Drop list to choose which is an active mask

            self.mask_choice_dropdown = QComboBox()
            self.mask_choice_dropdown.addItem('Instance Segmentation Mask', userData=0)
            self.mask_choice_dropdown.addItem('Labels Mask', userData=1)
            layout.addWidget(self.mask_choice_dropdown, 1, 1)



            # when user has chosen the mask, we don't want to change it anymore to avoid errors
            lock_button = QPushButton("Confirm Final Choice")
            lock_button.clicked.connect(self.set_active_mask)

            layout.addWidget(lock_button, 1, 2)
            layer.mouse_drag_callbacks.append(self.copy_mask_callback)
            layer.events.set_data.connect(lambda event: self.copy_mask_callback(layer, event))

       

        add_to_inprogress_button = QPushButton('Move to \'Curatation in progress\' folder')
        layout.addWidget(add_to_inprogress_button, 2, 0, 1, 2)
        add_to_inprogress_button.clicked.connect(self.on_add_to_inprogress_button_clicked)
    
        add_to_curated_button = QPushButton('Move to \'Curated dataset\' folder')
        layout.addWidget(add_to_curated_button, 2, 2, 1, 2)
        add_to_curated_button.clicked.connect(self.on_add_to_curated_button_clicked)

        self.setLayout(layout)

        # self.show()
    def set_active_mask(self):
        self.mask_choice_dropdown.setDisabled(True)
        self.active_mask_index = self.mask_choice_dropdown.currentData()

    def on_mask_choice_changed(self, index):
        self.active_mask_index = self.mask_choice_dropdown.itemData(index)

    def copy_mask_callback(self, layer, event):

        source_mask = layer.data

        if event.type == "mouse_press":

            c, event_x, event_y = event.position
            c, event_x, event_y = int(c), int(np.round(event_x)), int(np.round(event_y))
            self.event_coords = (c, event_x, event_y)

        elif event.type == "set_data":

            if self.event_coords is not None:
                c, event_x, event_y = self.event_coords
                
                if c == self.active_mask_index:

                    labels, counts = np.unique(source_mask[c, event_x - 1: event_x + 2, event_y - 1: event_y + 2], return_counts=True)
                    
                    if labels.size > 0:

                        idx = np.argmax(counts)
                        label = labels[idx]

                        mask_fill = source_mask[c] == label
                        source_mask[abs(c - 1)][mask_fill] = label

                        self.changed = True

                else:

                    mask_fill = source_mask[abs(c - 1)] == 0
                    source_mask[c][mask_fill] = 0


    def on_add_to_curated_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        if  self.app.cur_selected_path == str(self.app.train_data_path):
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
        self.app.move_images(self.app.train_data_path)

        # Save the (changed) seg
        self.app.save_image(self.app.train_data_path, cur_seg_selected+'.tiff', seg)

        # We remove seg from the current directory if it exists (both eval and inprogr allowed)
        self.app.delete_images(self.app.seg_filepaths)
        # TODO Create the Archive folder for the rest? Or move them as well? 

        self.viewer.close()
        self.close()

    def on_add_to_inprogress_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        # TODO: Do we allow this? What if they moved it by mistake? User can always manually move from their folders?)
        if self.app.cur_selected_path == str(self.app.train_data_path):
            message_text = "Images from '\Curated data'\ folder can not be moved back to \'Curatation in progress\' folder."
            utils.create_warning_box(message_text, message_title="Warning")
            return
        
        # take the name of the currently selected layer (by the user)
        cur_seg_selected = self.viewer.layers.selection.active.name
        # TODO if more than one item is selected this will break!
        if '_seg' not in cur_seg_selected:
            message_text = "Please select the segmenation you wish to save from the layer list"
            utils.create_warning_box(message_text, message_title="Warning")
            return

        # Move original image
        self.app.move_images(self.app.inprogr_data_path, move_segs=True)

        # Save the (changed) seg - this will overwrite existing seg if seg name hasn't been changed in viewer
        seg = self.viewer.layers[cur_seg_selected].data
        self.app.save_image(self.app.inprogr_data_path, cur_seg_selected+'.tiff', seg)
        
        self.close()