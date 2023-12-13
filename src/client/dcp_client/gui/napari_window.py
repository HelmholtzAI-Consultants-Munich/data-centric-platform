from __future__ import annotations
from typing import List, TYPE_CHECKING

from qtpy.QtWidgets import QWidget, QPushButton, QComboBox, QLabel, QGridLayout
from qtpy.QtCore import Qt
import napari

import numpy as np
from skimage.feature import canny
import cv2

if TYPE_CHECKING:
    from dcp_client.app import Application

from dcp_client.utils import utils

widget_list = [
    'ellipse_button',
    'line_button',
    'path_button',
    'polygon_button',
    'vertex_remove_button',
    'vertex_insert_button',
    'move_back_button',
    'move_front_button',
    'label_eraser',
]

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
        self.img = self.app.load_image()
        self.app.search_segs()

        # Set the viewer
        self.viewer = napari.Viewer(show=False)

        self.viewer.add_image(self.img, name=utils.get_path_stem(self.app.cur_selected_img))

        for seg_file in self.app.seg_filepaths:
            self.viewer.add_labels(self.app.load_image(seg_file), name=utils.get_path_stem(seg_file))

        self.layer = self.viewer.layers[utils.get_path_stem(self.app.seg_filepaths[0])]
        self.qctrl = self.viewer.window.qt_viewer.controls.widgets[self.layer]

        self.changed = False
        self.event_coords = None
        self.active_mask_instance = None

        main_window = self.viewer.window._qt_window
        
        layout = QGridLayout()
        layout.addWidget(main_window, 0, 0, 1, 4)

        # set first mask as active by default
        self.active_mask_index = 0

        # unique labels
        self.instances = set(np.unique(self.layer.data[self.active_mask_index])[1:])
        # for copying contours
        self.instances_updated = set()

        # For each instance find the contours and set the color of it to 0 to be invisible
        self.find_edges()
        # self.prev_mask = self.layer.data[0]

        self.switch_to_active_mask()

        if self.layer.data.shape[0] >= 2:
            # User hint
            message_label = QLabel('Choose an active mask')
            message_label.setAlignment(Qt.AlignRight)
            layout.addWidget(message_label, 1, 0)
        
        # Drop list to choose which is an active mask
            
            self.mask_choice_dropdown = QComboBox()
            self.mask_choice_dropdown.setEnabled(False)
            self.mask_choice_dropdown.addItem('Instance Segmentation Mask', userData=0)
            self.mask_choice_dropdown.addItem('Labels Mask', userData=1)
            layout.addWidget(self.mask_choice_dropdown, 1, 1)

            # when user has chosen the mask, we don't want to change it anymore to avoid errors
            lock_button = QPushButton("Confirm Final Choice")
            lock_button.setEnabled(False)
            lock_button.clicked.connect(self.set_active_mask)

            layout.addWidget(lock_button, 1, 2)
            self.layer.mouse_drag_callbacks.append(self.copy_mask_callback)
            self.layer.events.set_data.connect(lambda event: self.copy_mask_callback(self.layer, event))


        add_to_inprogress_button = QPushButton('Move to \'Curatation in progress\' folder')
        layout.addWidget(add_to_inprogress_button, 2, 0, 1, 2)
        add_to_inprogress_button.clicked.connect(self.on_add_to_inprogress_button_clicked)
    
        add_to_curated_button = QPushButton('Move to \'Curated dataset\' folder')
        layout.addWidget(add_to_curated_button, 2, 2, 1, 2)
        add_to_curated_button.clicked.connect(self.on_add_to_curated_button_clicked)

        self.setLayout(layout)

    def switch_controls(self, target_widget, status: bool):
        getattr(self.qctrl, target_widget).setEnabled(status)

    def switch_to_active_mask(self):

        self.switch_controls("paint_button", True)
        self.switch_controls("erase_button", True)
        self.switch_controls("fill_button", False)

        self.active_mask = True
    
    def switch_to_non_active_mask(self):

        self.instances = set(np.unique(self.layer.data[self.active_mask_index])[1:])

        self.switch_controls("paint_button", False)
        self.switch_controls("erase_button", False)
        self.switch_controls("fill_button", True) 

        self.active_mask = False

    def set_active_mask(self):
        self.mask_choice_dropdown.setDisabled(True)
        self.active_mask_index = self.mask_choice_dropdown.currentData()
        self.instances = set(np.unique(self.layer.data[self.active_mask_index])[1:])
        self.prev_mask = self.layer.data[self.active_mask]
        if self.active_mask_index == 1:
            self.switch_to_non_active_mask()

    def find_edges(self, idx=None):
        '''
        idx - indices of the specific labels from which to get contour
        '''
        if idx is not None and not isinstance(idx, list):
            idx = [idx]

        active_mask = self.layer.data[self.active_mask_index]

        instances = np.unique(active_mask)[1:]
        edges = np.zeros_like(active_mask).astype(int)

        # to merge the discontinuous contours
        kernel = np.ones((5, 5))

        if len(instances):
            for i in instances:
                if idx is None or i in idx:
        
                    mask_instance = (active_mask == i).astype(np.uint8)

                    edge_mask = 255 * (canny(255 * (mask_instance)) > 0).astype(np.uint8)
                    edge_mask = cv2.morphologyEx(
                        edge_mask, 
                        cv2.MORPH_CLOSE, 
                        kernel,
                    )
                    edges = edges + edge_mask

            # if masks are intersecting then we want to count it only once
            edges = edges > 0
            # cut the contours
            self.layer.data = self.layer.data * np.invert(edges).astype(np.uint8)
            
    def copy_mask_callback(self, layer, event):

        source_mask = layer.data
    
        if event.type == "mouse_press":

            c, event_x, event_y = event.position
            c, event_x, event_y = int(c), int(np.round(event_x)), int(np.round(event_y))

            if source_mask[c, event_x, event_y] == 0:
                self.new_pixel = True
            else:
                self.new_pixel = False

            self.event_coords = (c, event_x, event_y)

           

        elif event.type == "set_data":
            
            active_mask_current = self.active_mask

            if self.viewer.dims.current_step[0] == self.active_mask_index:
                self.switch_to_active_mask()
            else:
                self.switch_to_non_active_mask()

            if self.event_coords is not None:

                c, event_x, event_y = self.event_coords
                
                if c == self.active_mask_index:

                    labels, counts = np.unique(source_mask[c, event_x - 1: event_x + 2, event_y - 1: event_y + 2], return_counts=True)

                    if labels.size > 0:
                     
                        idx = np.argmax(counts)
                        label = labels[idx]

                        mask_fill = source_mask[c] == label

                        # self.changed = True
                        # self.instances_updated.add(label)

                        # Find the color of the label mask at the given point
                        labels_seg, counts_seg = np.unique(
                            source_mask[abs(c - 1)][mask_fill], 
                            return_counts=True
                        )
                        idx_seg = np.argmax(counts_seg)
                        label_seg = labels_seg[idx_seg]

                        # If a new color is used, then it is copied to a label mask
                        # Otherwise, we copy the existing color from the label mask 
                        
                        if not label in self.instances:
                            source_mask[abs(c - 1)][mask_fill] = label
                        else:
                            source_mask[abs(c - 1)][mask_fill] = label_seg

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
        
        self.viewer.close()
        self.close()