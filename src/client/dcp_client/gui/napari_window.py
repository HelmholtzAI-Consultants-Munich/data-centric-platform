from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from copy import deepcopy

from qtpy.QtWidgets import (
    QPushButton, QComboBox, QLabel, QGridLayout, QWidget, 
    QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout
)
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QGuiApplication
import napari
import numpy as np

if TYPE_CHECKING:
    from dcp_client.app import Application
from dcp_client.utils.utils import get_path_stem, check_equal_arrays
from dcp_client.utils.compute4mask import Compute4Mask
from dcp_client.gui._my_widget import MyWidget
from dcp_client.gui.sam_controller import SAMController


class NapariWindow(MyWidget):
    """Napari Window Widget object.
    Opens the napari image viewer to view and fix the labeles.
    :param app: The Application instance.
    :type app: Application
    """

    def __init__(self, app: Application) -> None:
        """Initializes the NapariWindow.

        :param app: The Application instance.
        :type app: Application
        """
        super().__init__()
        self.app = app
        self.setWindowTitle("napari viewer")
        self.setStyleSheet("background-color: #262930;")
        screen_size = QGuiApplication.primaryScreen().geometry()
        self.resize(int(screen_size.width()*0.8), int(screen_size.height()*0.8))

        self.fix_small_objs = True  # flag to control whether to fix small objects or not
        self.fix_small_holes = True # flag to control whether to fix small holes or not
        # Load image and get corresponding segmentation filenames
        img = self.app.load_image()
        self.app.search_segs()
        self.seg_files = self.app.seg_filepaths.copy()

        # Set the viewer
        self.viewer = napari.Viewer(show=False)

        # Initialize qctrl and mask_choice_dropdown early to ensure they exist as attributes
        # They may remain None if the multi-channel segmentation condition is not met
        self.qctrl = None
        self.mask_choice_dropdown = None
        
        # SAM controller (lazy initialized when assisted labelling is enabled)
        self.sam_controller: Optional[SAMController] = None

        self.viewer.add_image(img, name=get_path_stem(self.app.cur_selected_img))
        for seg_file in self.seg_files:
            seg = self.app.load_image(seg_file)
            seg = seg.astype(np.int16)  # fix dtype issue
            # check the masks of the loaded segmentation
            if seg.ndim>2 and self.app.num_classes==1:
                message_text = (
                    "Your mask contains a channel for class labels, but you specified only one class at runtime.\n"
                    + "Click 'Cancel' to go back and re-run the segmentation algorithm to get only instance segmentation masks.\n"
                    + "If you like to proceed click 'Ok' and the additional class label will be removed automatically."
                )
                usr_response = self.create_selection_box(message_text, "Additional mask channel found!")
                if usr_response=='action': 
                    seg = seg[0]
                else:
                    print('User cancelled due to extra mask channel. Closing viewer.')
                    QTimer.singleShot(0, self.close)
                    return
                
            elif seg.ndim==2 and self.app.num_classes>1:
                seg = Compute4Mask.add_seg_layer(seg)
            self.viewer.add_labels(seg, name=get_path_stem(seg_file))

        main_window = self.viewer.window._qt_window
        # main_window.setFixedSize(1200,600)
        layout = QGridLayout()
        layout.addWidget(main_window, 0, 0, 1, 3)

        # Assisted Labelling Control Panel
        assisted_labelling_widget = QWidget()
        assisted_labelling_layout = QHBoxLayout()
        assisted_labelling_widget.setStyleSheet("background-color: #262930; padding: 10px;")
        assisted_labelling_layout.addStretch(1) # Add stretch on the left to push content to the right

        # Left side: Toggle button for Assisted Labelling
        toggle_label = QLabel("Assisted labelling")
        toggle_label.setStyleSheet("color: #D1D2D4; font-size: 12px; font-weight: bold;")
        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.setMaximumWidth(50)
        self.toggle_button.setStyleSheet(
            """
            QPushButton {
                background-color: #5A626C;
                color: #D1D2D4;
                border-radius: 3px;
                padding: 2px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #0064A8;
            }
            """
        )
        self.toggle_button.setText("OFF")
        
        def update_toggle_text(checked):
            self.toggle_button.setText("ON" if checked else "OFF")
        
        self.toggle_button.toggled.connect(update_toggle_text)
        self.toggle_button.toggled.connect(self.on_assisted_labelling)
        #self.toggle_button.setStyleSheet("color: #D1D2D4; font-size: 12px; font-weight: bold;")
        #self.toggle_button.setToolTip("Enable or disable assisted labelling mode")

        left_layout = QHBoxLayout()
        left_layout.addWidget(toggle_label)
        left_layout.addWidget(self.toggle_button)
        left_layout.addStretch()

        # Right side: Prompting radio buttons
        prompting_label = QLabel("Prompting")
        prompting_label.setStyleSheet("color: #D1D2D4; font-size: 12px; font-weight: bold;")

        self.prompting_group = QButtonGroup()
        boxes_radio = QRadioButton("Boxes")
        points_radio = QRadioButton("Points")

        boxes_radio.setStyleSheet("color: #D1D2D4; font-size: 12px; font-weight: bold;")
        points_radio.setStyleSheet("color: #D1D2D4; font-size: 12px; font-weight: bold;")

        boxes_radio.setToolTip("Draw bounding boxes around objects to prompt the model. Press enter to accept the boxes, or escape to reject the last box.")
        point_hover_text = ("Click points inside objects to prompt the model. Toggle between foreground (white) and background (red) point mode with the 'b' key. Once you are done click the 'd' key on your"
                            " keyboard to get the result!")
        points_radio.setToolTip(point_hover_text)

        self.prompting_group.addButton(boxes_radio, 0)
        self.prompting_group.addButton(points_radio, 1)
        boxes_radio.setChecked(True)  # Default to boxes
        self.prompting_group.idClicked.connect(self.on_prompting_mode_changed)

        right_layout = QHBoxLayout()
        right_layout.addWidget(prompting_label)
        right_layout.addWidget(boxes_radio)
        right_layout.addWidget(points_radio)
        right_layout.addStretch()

        # Bottom right: Hover explanation text
        hover_help = QLabel("Hover over the prompting options for explanations on how to use!")
        hover_help.setStyleSheet(
            "color: #A0A2A4; font-size: 10px; font-style: italic; "
            "background-color: transparent;"
        )
        hover_help_layout = QHBoxLayout()
        hover_help_layout.addStretch()
        hover_help_layout.addWidget(hover_help)

        # Combine layouts
        assisted_labelling_layout.addLayout(left_layout, 1)
        assisted_labelling_layout.addLayout(right_layout, 1)
        assisted_labelling_widget.setLayout(assisted_labelling_layout)

        # Add tooltip widget below
        tooltip_widget = QWidget()
        tooltip_layout = QVBoxLayout()
        tooltip_layout.addWidget(assisted_labelling_widget)
        tooltip_layout.addLayout(hover_help_layout)
        tooltip_layout.setContentsMargins(0, 0, 10, 5)
        tooltip_widget.setLayout(tooltip_layout)

        layout.addWidget(tooltip_widget, 1, 0, 1, 3)

        # select the first seg as the currently selected layer if there are any segs
        if (
            len(self.seg_files)
            and len(self.viewer.layers[get_path_stem(self.seg_files[0])].data.shape) > 2
        ):
            self.cur_selected_seg = self.viewer.layers.selection.active.name
            self.layer = self.viewer.layers[self.cur_selected_seg]
            self.viewer.layers.selection.events.changed.connect(
                self.on_seg_channel_changed
            )
            # set first mask as active by default
            self.active_mask_index = 0
            self.viewer.dims.events.current_step.connect(self.axis_changed)
            self.original_instance_mask = {}
            self.original_class_mask = {}
            self.instances = {}
            self.contours_mask = {}
            for seg_file in self.seg_files:
                layer_name = get_path_stem(seg_file)
                # get unique instance labels for each seg
                self.original_instance_mask[layer_name] = deepcopy(
                    self.viewer.layers[layer_name].data[0]
                )
                self.original_class_mask[layer_name] = deepcopy(
                    self.viewer.layers[layer_name].data[1]
                )
                # compute unique instance ids
                self.instances[layer_name] = Compute4Mask.get_unique_objects(
                    self.original_instance_mask[layer_name]
                )
                # remove border from class mask
                self.contours_mask[layer_name] = Compute4Mask.get_contours(
                    self.original_instance_mask[layer_name], contours_level=0.8
                )
                self.viewer.layers[layer_name].data[1][
                    self.contours_mask[layer_name] != 0
                ] = 0

            self.qctrl = self.viewer.window.qt_viewer.controls.widgets[self.layer]

            if len(self.layer.data.shape) > 2:
                # User hint
                message_label = QLabel('Choose an active mask')
                message_label.setStyleSheet(
                """
                    font-size: 12px;
                    font-weight: bold; 
                    background-color: #262930;
                    color: #D1D2D4;
                    border-radius: 5px; 
                    padding: 8px 16px;"""
                )

                message_label.setAlignment(Qt.AlignRight)
                layout.addWidget(message_label, 2, 0)

                # Drop list to choose which is an active mask
                self.mask_choice_dropdown = QComboBox()
                self.mask_choice_dropdown.setEnabled(False)
                self.mask_choice_dropdown.addItem(
                    "Instance Segmentation Mask", userData=0
                )
                self.mask_choice_dropdown.addItem("Labels Mask", userData=1)
                layout.addWidget(self.mask_choice_dropdown, 2, 1)

                # when user has chosen the mask, we don't want to change it anymore to avoid errors
                lock_button = QPushButton("Confirm Final Choice")
                lock_button.setStyleSheet(
                """QPushButton 
                        { 
                            background-color: #5A626C;
                            font-size: 12px; 
                            font-weight: bold;
                            color: #D1D2D4; 
                            border-radius: 5px;
                            padding: 8px 16px; }"""
                "QPushButton:hover { background-color: #6A7380; }"
                "QPushButton:pressed { background-color: #6A7380; }" 
                    )

                lock_button.setEnabled(True)
                lock_button.clicked.connect(self.set_editable_mask)

                layout.addWidget(lock_button, 2, 2)
        else:
            self.layer = None

        # add buttons for moving images to other dirs
        add_to_inprogress_button = QPushButton('Move to \'Curatation in progress\' folder')
        add_to_inprogress_button.setStyleSheet(
        """QPushButton 
            { 
                  background-color: #0064A8;
                  font-size: 12px; 
                  font-weight: bold;
                  color: #D1D2D4; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
        "QPushButton:hover { background-color: #006FBA; }"
        "QPushButton:pressed { background-color: #006FBA; }" 
               
         
        )
        layout.addWidget(add_to_inprogress_button, 3, 0)#, 1, 2)
        add_to_inprogress_button.clicked.connect(
            self.on_add_to_inprogress_button_clicked
        )

        add_to_curated_button = QPushButton('Move to \'Curated dataset\' folder')
        add_to_curated_button.setStyleSheet(
        """QPushButton 
            { 
                  background-color: #0064A8;
                  font-size: 12px; 
                  font-weight: bold;
                  color: #D1D2D4; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
        "QPushButton:hover { background-color: #006FBA; }"
        "QPushButton:pressed { background-color: #006FBA; }" 
         
        )

        layout.addWidget(add_to_curated_button, 3, 1)#, 1, 2)
        add_to_curated_button.clicked.connect(self.on_add_to_curated_button_clicked)

        self.setLayout(layout)

        remove_from_dataset_button = QPushButton('Remove image from dataset')
        remove_from_dataset_button.setStyleSheet(
        """QPushButton 
            { 
                  background-color: #0064A8;
                  font-size: 12px; 
                  font-weight: bold;
                  color: #D1D2D4; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
        "QPushButton:hover { background-color: #006FBA; }"
        "QPushButton:pressed { background-color: #006FBA; }" 
         
        )
        layout.addWidget(remove_from_dataset_button, 3, 2)
        remove_from_dataset_button.clicked.connect(self.on_remove_from_dataset_button_clicked)

        QTimer.singleShot(0, self.adjust_window_size)
        QTimer.singleShot(0, lambda: self.viewer.reset_view())
        self.show()

    def closeEvent(self, event):
        """Properly close Napari viewer and all child widgets."""
        try:
            # Stop SAM worker thread (signal only, no blocking wait)
            if self.sam_controller:
                self.sam_controller.stop()
                self.sam_controller = None
            
            # Close Napari viewer if exists
            if hasattr(self, 'viewer') and self.viewer is not None:
                try:
                    if self.viewer.window is not None:
                        # Close any dock widgets
                        try:
                            for dock in self.viewer.window._dock_widgets.values():
                                try:
                                    dock.close()
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Try to close the napari window and viewer
                        try:
                            self.viewer.window.close()
                        except Exception:
                            pass
                except Exception:
                    pass
                # Clear the viewer reference
                self.viewer = None

            # Close any other child widgets
            for child in self.findChildren(QWidget):
                if child is not self:
                    try:
                        child.close()
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error during closeEvent cleanup: {e}")

        # Always accept the close event
        event.accept()

    def adjust_window_size(self):
        """Resize Napari main window to 80% of available screen, centered."""
        window = self.viewer.window._qt_window #self.viewer.window._qt_window
        screen = QGuiApplication.screenAt(window.pos()) or QGuiApplication.primaryScreen()
        avail = screen.availableGeometry()
        w = int(avail.width() * 0.8)
        h = int(avail.height() * 0.8)
        x = avail.x() + (avail.width() - w) // 2
        y = avail.y() + (avail.height() - h) // 2
        window.setGeometry(x, y, w, h)

    def on_remove_from_dataset_button_clicked(self) -> None:
        """
        Defines what happens when the "Remove from dataset" button is clicked.
        """
        '''
        try:
            # get image name
            files_to_remove = [self.viewer.layers.selection.active.name]
        except AttributeError:
            message_text = "Please first select the image in the layer list."
            _ = self.create_warning_box(message_text, message_title="Warning")
            return
        '''
        rmv_files = [self.app.cur_selected_img]
        self.app.search_segs()
        rmv_files.extend(self.app.seg_filepaths)
        # Delete the image and corresponding masks from the dataset
        self.app.delete_images(rmv_files)
        self.app.cur_selected_img = ""
        self.app.seg_filepaths = []
        self.viewer.close()
        self.close()

    def set_editable_mask(self) -> None:
        """
        This function is not implemented. In theory the use can choose between which mask to edit.
        Currently painting and erasing is only possible on instance mask and in the class mask only
        the class labels can be changed.
        """
        pass

    def on_assisted_labelling(self, checked: bool) -> None:
        """
        Triggered when the user toggles the Assisted labelling mode ON or OFF.
        """
        if not checked:
            if self.sam_controller:
                self.sam_controller.disable()
            try:
                self.viewer.bind_key('Enter', None, overwrite=True)
                self.viewer.bind_key('Escape', None, overwrite=True)
            except Exception:
                pass
            return

        # Initialize SAM controller if needed
        needs_image_set = False
        if not self.sam_controller:
            self.sam_controller = SAMController(self.viewer, self.app)
            needs_image_set = True
        
        if not self.sam_controller.initialize():
            self.toggle_button.setChecked(False)
            return
        
        if needs_image_set:
            self.sam_controller.set_image()
        
        def bind_viewer_keys():
            try:
                self.viewer.bind_key('Enter', lambda v: self.sam_controller.accept_masks(), overwrite=True)
                self.viewer.bind_key('Escape', lambda v: self.sam_controller.reject_last_mask(), overwrite=True)
            except Exception:
                pass
        
        mode = self.prompting_group.checkedId()
        self.sam_controller.enable(mode, bind_viewer_keys)

    def on_prompting_mode_changed(self, _id: int) -> None:
        """
        Triggered when the user switches between 'Boxes' and 'Points'.
        """
        if hasattr(self, 'toggle_button') and self.toggle_button.isChecked():
            self.on_assisted_labelling(True)


    def on_seg_channel_changed(self, event) -> None:
        """
        Is triggered each time the user selects a different layer in the viewer.
        """
        if (act := self.viewer.layers.selection.active) is not None:
            # Only update cur_selected_seg for actual segmentation layers (not SAM layers)
            if isinstance(act, napari.layers.Labels) and "_seg" in act.name:
                self.cur_selected_seg = act.name
                self.layer = act
            elif isinstance(act, napari.layers.Image):
                pass
        else:
            pass

    def axis_changed(self, event) -> None:
        """
        Is triggered each time the user switches the viewer between the mask channels. At this point the class mask
        needs to be updated according to the changes made to the instance segmentation mask.
        """
        if self.app.cur_selected_img=="": return # because this also gets triggered when removing outlier

        self.active_mask_index = self.viewer.dims.current_step[0]
        
        # Ensure self.layer points to the correct segmentation layer
        if hasattr(self, 'cur_selected_seg') and self.cur_selected_seg:
            try:
                self.layer = self.viewer.layers[self.cur_selected_seg]
            except KeyError:
                return
        
        try:
            masks = deepcopy(self.layer.data)
        except Exception:
            return

        # if user has switched to the instance mask
        if self.active_mask_index == 0:
            class_mask_with_contours = Compute4Mask.add_contour(masks[1], masks[0])
            if not check_equal_arrays(
                class_mask_with_contours.astype(bool),
                self.original_class_mask[self.cur_selected_seg].astype(bool),
            ):
                self.update_instance_mask(masks[0], masks[1])
            self.switch_to_instance_mask()

        # else if user has switched to the class mask
        elif self.active_mask_index == 1:
            if not check_equal_arrays(
                masks[0], self.original_instance_mask[self.cur_selected_seg]
            ):
                self.update_labels_mask(masks[0])
            self.switch_to_labels_mask()

    def switch_to_instance_mask(self) -> None:
        """
        Switch the application to the active mask mode by enabling 'paint_button', 'erase_button'
        and 'fill_button'.
        """
        
        self.original_class_mask[self.cur_selected_seg] = deepcopy(self.layer.data[1])
        self.switch_controls("paint_button", True)
        self.switch_controls("erase_button", True)
        self.switch_controls("fill_button", True)
        
        # Re-enable SAM toggle on instance channel
        if hasattr(self, 'toggle_button'):
            self.toggle_button.setEnabled(True)
            self.toggle_button.setToolTip("")
            # Restore SAM if it was on before channel switch
            if getattr(self, '_sam_was_enabled_before_channel_switch', False):
                self._sam_was_enabled_before_channel_switch = False
                self.toggle_button.setChecked(True)

    def switch_to_labels_mask(self) -> None:
        """
        Switch the application to non-active mask mode by enabling 'fill_button' and disabling 'paint_button' and 'erase_button'.
        """

        self.original_instance_mask[self.cur_selected_seg] = deepcopy(self.layer.data[0])
        
        # Remove SAM layers when switching to class channel
        if hasattr(self, 'toggle_button') and self.toggle_button.isChecked():
            self._sam_was_enabled_before_channel_switch = True
            self.toggle_button.blockSignals(True)
            self.toggle_button.setChecked(False)
            self.toggle_button.blockSignals(False)
        
        for lname in ("box-prompts", "point-prompts", "sam-preview"):
            if lname in self.viewer.layers:
                self.viewer.layers.remove(lname)
        
        # Disable SAM toggle on class channel
        if hasattr(self, 'toggle_button'):
            self.toggle_button.setEnabled(False)
            self.toggle_button.setToolTip("SAM is only available on Instance channel")
        
        # Select segmentation layer and set fill mode
        if self.cur_selected_seg in [layer.name for layer in self.viewer.layers]:
            seg_layer = self.viewer.layers[self.cur_selected_seg]
            self.viewer.layers.selection.clear()
            self.viewer.layers.selection.add(seg_layer)
            seg_layer.mode = "fill"
            
        info_message_paint = (
            "Painting objects is only possible in the instance layer for now."
        )
        info_message_erase = (
            "Erasing objects is only possible in the instance layer for now."
        )
        self.switch_controls("paint_button", False, info_message_paint)
        self.switch_controls("erase_button", False, info_message_erase)
        self.switch_controls("fill_button", True)

    def update_labels_mask(self, instance_mask: np.ndarray) -> None:
        """Updates the class mask based on changes in the instance mask.

        If the instance mask has changed since the last switch between channels, the class mask needs to be updated accordingly.

        :param instance_mask: The updated instance mask, changed by the user.
        :type instance_mask: numpy.ndarray
        :return: None
        """
        self.original_class_mask[self.cur_selected_seg] = (
            Compute4Mask.compute_new_labels_mask(
                self.original_class_mask[self.cur_selected_seg],
                instance_mask,
                self.original_instance_mask[self.cur_selected_seg],
                self.instances[self.cur_selected_seg],
            )
        )
        # update original instance mask and instances
        self.original_instance_mask[self.cur_selected_seg] = instance_mask
        self.instances[self.cur_selected_seg] = Compute4Mask.get_unique_objects(
            self.original_instance_mask[self.cur_selected_seg]
        )
        # compute contours to remove from class mask visualisation
        self.contours_mask[self.cur_selected_seg] = Compute4Mask.get_contours(
            instance_mask, contours_level=0.8
        )
        vis_labels_mask = deepcopy(self.original_class_mask[self.cur_selected_seg])
        vis_labels_mask[self.contours_mask[self.cur_selected_seg] != 0] = 0
        # update the viewer
        self.layer.data[1] = vis_labels_mask
        self.layer.refresh()

    def update_instance_mask(
        self, instance_mask: np.ndarray, labels_mask: np.ndarray
    ) -> None:
        """Updates the instance mask based on changes in the labels mask.

        If the labels mask has changed, but only if an object has been removed, the instance mask is updated accordingly.

        :param instance_mask: The existing instance mask, which needs to be updated.
        :type instance_mask: numpy.ndarray
        :param labels_mask: The updated labels mask, changed by the user.
        :type labels_mask: numpy.ndarray
        """
        
        # add contours back to labels mask
        labels_mask = Compute4Mask.add_contour(labels_mask, instance_mask)
        # and compute the updated instance mask
        self.original_instance_mask[self.cur_selected_seg] = (
            Compute4Mask.compute_new_instance_mask(labels_mask, instance_mask)
        )
        self.instances[self.cur_selected_seg] = Compute4Mask.get_unique_objects(
            self.original_instance_mask[self.cur_selected_seg]
        )
        self.original_class_mask[self.cur_selected_seg] = labels_mask
        # update the viewer
        self.layer.data[0] = self.original_instance_mask[self.cur_selected_seg]
        self.layer.refresh()

    def switch_controls(
        self, target_widget: str, status: bool, info_message: Optional[str] = None
    ) -> None:
        """Enables or disables a specific widget.

        :param target_widget: The name of the widget to be controlled within the QCtrl object.
        :type target_widget: str
        :param status: If True, the widget will be enabled; if False, it will be disabled.
        :type status: bool
        :param info_message: Optionally add an info message when hovering over some widget. Default is None.
        :type info_message: str or None
        """
        try:
            getattr(self.qctrl, target_widget).setEnabled(status)
            if info_message is not None:
                getattr(self.qctrl, target_widget).setToolTip(info_message)
        except:
            pass

    def check_and_update_if_layers_changed(self, seg, class_mask_with_contour, seg_name_to_save):
        """
        Checks to see if changes have been made to the layers right before saving. 
        Updates the masks if so. This function handles the case when e.g. user deletes
        an object from the instance mask and then directly tries to move the data to in 
        progress. The class label needs to be updated with the change.
        :param seg: Current seg layers
        :type seg: List
        :param class_mask_with_contour: Class mask with contour added
        :type class_mask_with_contour: np.ndarray
        :param seg_name_to_save: Name of the segmentation layer user wants to save
        :type seg_name_to_save: str
        """
        if not check_equal_arrays(
                seg[0].astype(bool),
                class_mask_with_contour.astype(bool)
                ):
            print('Updating masks before saving...')
            if self.active_mask_index==1: self.update_instance_mask(seg[0], seg[1])
            else: self.update_labels_mask(seg[0])
            # reload the seg layers after update
            seg = self.viewer.layers[seg_name_to_save].data
            seg[1] = Compute4Mask.add_contour(seg[1], seg[0])
        return seg

    def on_save_to_folder_clicked(self, save_folder, move_segs=False) -> None:
        """ Is called when either of the two buttons are clicked to save data to new folder
        :param save_folder: Indicates the directory where we wish to save the data
        :type save_folder: str
        :param move_segs: Indicates whether all labels layers found in same directory with the image should be moved too. 
                            If we are moving to in_progress directory all segs are moved, otherwise only the one selected by the user.
        :type move_segs: bool
        """
         # TODO: Do we allow this? What if they moved it by mistake? User can always manually move from their folders?)
        # check if user is trying to save image which is already in curated folder - not allowed to change!
        if self.app.cur_selected_path == str(self.app.train_data_path):
            message_text = "Image is already in the 'Curated data' folder and should not be changed again"
            _ = self.create_warning_box(message_text, message_title="Warning")
            return

        if self.app.cur_selected_path == save_folder:
            message_text = "Image is already in the 'In progress data' folder - Did you mean to move it to the 'Curated data' folder? Go back and try again!"
            _ = self.create_warning_box(message_text, message_title="Warning")
            return
        
        # take the name of the currently selected layer (by the user)
        seg_name_to_save = self.viewer.layers.selection.active.name
        # TODO if more than one item is selected this will break!
        if "_seg" not in seg_name_to_save:
            message_text = (
                "Please select the segmenation you wish to save from the layer list."
                "The labels layer should have the same name as the image to which it corresponds, followed by _seg."
            )
            _ = self.create_warning_box(message_text, message_title="Warning")
            return
        
        # get the labels layer
        seg = self.viewer.layers[seg_name_to_save].data
        
        annot_error, faulty_ids_annot = Compute4Mask.assert_connected_objects(seg)

        if annot_error:
            message_text = (
                "There seems to be a problem with your mask. We expect each object to be a connected component. For object(s) with ID(s): \n"
                + ", ".join(str(id) for id in faulty_ids_annot[:-1])
                + (", " if len(faulty_ids_annot) > 1 else "")
                + str(faulty_ids_annot[-1])
                + " more than one connected component was found. Would you like us to clean this up and keep only the largest connect component?"
            )
            usr_response = self.create_selection_box(message_text, "Clean up")
            print('User response to connected components check:', usr_response)
            if usr_response=='action' and self.fix_small_objs: 
                seg = Compute4Mask.keep_largest_components_pair(seg, faulty_ids_annot)
                self.viewer.layers[seg_name_to_save].data = seg
                self.viewer.layers[seg_name_to_save].refresh()
            else: 
                self.fix_small_objs = False
                
        print('Connected component checks passed.')

        annot_error, holes = Compute4Mask.assert_filled_objects(seg)
        print('Holes found in objects:', annot_error)
        if annot_error:
            message_text = (
                "For object(s) with ID(s): "+ ", ".join(str(id) for id in list(holes.keys())[:-1])
                + (", " if len(list(holes.keys())) > 1 else "")
                + str(list(holes.keys())[-1])
                + " holes where found. Would you like us to clean this up and fill the holes in the segmentation?"
            )
            usr_response = self.create_selection_box(message_text, "Clean up")
            print('User response to holes check:', usr_response)
            if usr_response=='action' and self.fix_small_holes: 
                seg = Compute4Mask.fill_holes(seg, holes)
                self.viewer.layers[seg_name_to_save].data = seg
                self.viewer.layers[seg_name_to_save].refresh()
            else:
                self.fix_small_holes = False 
                
        print('Objects with holes checks passed.')

        if self.app.num_classes>1:

            class_mask_with_contour = Compute4Mask.add_contour(seg[1], seg[0]) # add contour to labels mask
            seg = self.check_and_update_if_layers_changed(seg, class_mask_with_contour, seg_name_to_save)

            annot_error = Compute4Mask.assert_missing_classes(seg)
            if annot_error:
                inst= seg[0]
                message_text = (
                    "You still haven't annotated all obects in your class mask. Please go back and complete the annotation, replacing" \
                    + " any objects with default value '-1' with the actual class label. Instance id "+ str(np.unique(inst[np.where(seg[1]==-1)]) ) +" is still unlabelled."
                )
                usr_response = self.create_selection_box(message_text, "Annotation incomplete!")
                return
        
            annot_error = Compute4Mask.assert_num_classes(seg, self.app.num_classes)
            if annot_error: 
                class_ids = Compute4Mask.get_unique_objects(seg[1])
                message_text = (
                    "Your class label contains "+ str(len(class_ids)) +" classses, whereas you specified "
                    + str(self.app.num_classes) + " number of classes at runtime."
                    + "Please go back and fix your class mask. Current class ids in your image are: " 
                    + ", ".join(str(id) for id in class_ids)
                )
                usr_response = self.create_selection_box(message_text, "Annotation incomplete!")
                return
        print('Annotation checks passed.')

        # Move original image
        self.app.move_images(save_folder, move_segs)

        # Save the (changed) seg
        self.app.save_image(
            save_folder, seg_name_to_save + ".tiff", seg
        )
        # We remove segs from the current directory if it exists (both eval and inprogr allowed)
        self.app.delete_images(self.seg_files)

        self.viewer.close()
        self.close()

    def on_add_to_curated_button_clicked(self) -> None:
        """Defines what happens when the "Move to curated dataset folder" button is clicked."""
        self.on_save_to_folder_clicked(self.app.train_data_path)
    
    def on_add_to_inprogress_button_clicked(self) -> None:
        """Defines what happens when the "Move to curation in progress folder" button is clicked."""
        self.on_save_to_folder_clicked(self.app.inprogr_data_path, move_segs=True)
