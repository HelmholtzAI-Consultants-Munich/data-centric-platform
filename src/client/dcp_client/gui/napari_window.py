from __future__ import annotations
from typing import List, TYPE_CHECKING

from qtpy.QtWidgets import QWidget, QPushButton, QComboBox, QLabel, QGridLayout
from qtpy.QtCore import Qt
import napari
from napari.qt import thread_worker
from dcp_client.utils.utils import Compute4Mask


if TYPE_CHECKING:
    from dcp_client.app import Application

from dcp_client.utils.utils import get_path_stem
from dcp_client.gui._my_widget import MyWidget

class NapariWindow(MyWidget):
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
        self.viewer.add_image(img, name=get_path_stem(self.app.cur_selected_img))
        for seg_file in self.app.seg_filepaths:
            self.viewer.add_labels(self.app.load_image(seg_file), name=get_path_stem(seg_file))

        self.layer = self.viewer.layers[get_path_stem(self.app.seg_filepaths[0])]
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
        self.instances = Compute4Mask.get_unique_objects(self.layer.data[self.active_mask_index])
        
        # for copying contours
        self.instances_updated = set()

        # For each instance find the contours and set the color of it to 0 to be invisible
        edges = Compute4Mask.find_edges(instance_mask=self.layer.data[0])
        self.layer.data = self.layer.data * (~edges).astype(int)

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

            # when user has chosens the mask, we don't want to change it anymore to avoid errors
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
        """
        Enable or disable a specific widget.

        Parameters:
        - target_widget (str): The name of the widget to be controlled within the QCtrl object.
        - status (bool): If True, the widget will be enabled; if False, it will be disabled.
    
        """
        getattr(self.qctrl, target_widget).setEnabled(status)

    def switch_to_active_mask(self):
        """
        Switch the application to the active mask mode by enabling 'paint_button', 'erase_button' 
        and 'fill_button'.
        """

        self.switch_controls("paint_button", True)
        self.switch_controls("erase_button", True)
        self.switch_controls("fill_button", True)

        self.active_mask = True
    
    def switch_to_non_active_mask(self):
        """
        Switch the application to non-active mask mode by enabling 'fill_button' and disabling 'paint_button' and 'erase_button'.
        """

        self.instances = Compute4Mask.get_unique_objects(self.layer.data[self.active_mask_index])


        self.switch_controls("paint_button", False)
        self.switch_controls("erase_button", False)
        self.switch_controls("fill_button", True) 

        self.active_mask = False

    def set_active_mask(self):
        """
        Sets the active mask index based on the drop down list, by default 
        instance segmentation mask is an active mask with index 0.
        If the active mask index is 1, it switches to non-active mask mode.
        """
        if self.active_mask_index == 1:
            self.switch_to_non_active_mask()

   
    def copy_mask_callback(self, layer, event):
        """
        Handles mouse press and set data events to copy masks based on the active mask index.
        Parameters:
            - layer: The layer object associated with the mask.
            - event: The event triggering the callback.
        """

        source_mask = layer.data
    
        if event.type == "mouse_press":

            c, event_x, event_y = Compute4Mask.get_rounded_pos(event.position)
            self.event_coords = (c, event_x, event_y)

           
        elif event.type == "set_data":
            
            if self.viewer.dims.current_step[0] == self.active_mask_index:
                self.switch_to_active_mask()
            else:
                self.switch_to_non_active_mask()

            if self.event_coords is not None:

                c, event_x, event_y = self.event_coords
                
                if c == self.active_mask_index:
                    
                    # When clicking, the mouse provides a continuous position.
                    # To identify the color placement, we examine nearby positions within one pixel [idx_x - 1, idx_x + 1] and [idx_y - 1, idx_y + 1].

                    labels, counts = Compute4Mask.get_unique_counts_around_event(source_mask, c, event_x, event_y)

                    if labels.size > 0:
                     
                        # index of the most common color in the area around the click excluding 0 
                        idx = Compute4Mask.argmax(counts)
                        # the most common color in the area around the click 
                        label = labels[idx]
                        # get the mask of the instance
                        mask_fill = source_mask[c] == label

                        # Find the color of the label mask at the given point
                        # Determine the most common color in the label mask
                        labels_seg, counts_seg = Compute4Mask.get_unique_counts_for_mask(source_mask, c, mask_fill)
                        idx_seg = Compute4Mask.argmax(counts_seg)
                        label_seg = labels_seg[idx_seg]

                        # If a new color is used, then it is copied to a label mask
                        # Otherwise, we copy the existing color from the label mask 
                        
                        if not label in self.instances:
                            source_mask[abs(c - 1)][mask_fill] = label
                        else:
                            source_mask[abs(c - 1)][mask_fill] = label_seg

                else:
                    # the only action to be applied to the instance mask is erasing.
                    mask_fill = source_mask[abs(c - 1)] == 0
                    source_mask[c][mask_fill] = 0


    def on_add_to_curated_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        if  self.app.cur_selected_path == str(self.app.train_data_path):
            message_text = "Image is already in the \'Curated data\' folder and should not be changed again"
            _ = self.create_warning_box(message_text, message_title="Warning")
            return
        
        # take the name of the currently selected layer (by the user)
        cur_seg_selected = self.viewer.layers.selection.active.name
        # TODO if more than one item is selected this will break!
        if '_seg' not in cur_seg_selected:
            message_text = "Please select the segmenation you wish to save from the layer list"
            _ = self.create_warning_box(message_text, message_title="Warning")
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
            _ = self.create_warning_box(message_text, message_title="Warning")
            return
        
        # take the name of the currently selected layer (by the user)
        cur_seg_selected = self.viewer.layers.selection.active.name
        # TODO if more than one item is selected this will break!
        if '_seg' not in cur_seg_selected:
            message_text = "Please select the segmenation you wish to save from the layer list"
            _ = self.create_warning_box(message_text, message_title="Warning")
            return

        # Move original image
        self.app.move_images(self.app.inprogr_data_path, move_segs=True)

        # Save the (changed) seg - this will overwrite existing seg if seg name hasn't been changed in viewer
        seg = self.viewer.layers[cur_seg_selected].data
        self.app.save_image(self.app.inprogr_data_path, cur_seg_selected+'.tiff', seg)
        
        self.viewer.close()
        self.close()