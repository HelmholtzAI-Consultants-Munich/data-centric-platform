from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from qtpy.QtWidgets import (
                            QPushButton,
                            QVBoxLayout,
                            QHBoxLayout,
                            QSizePolicy,
                           QWidget,
                            QLabel,
                            QTreeView,
                            QProgressBar,
                            QShortcut,
                            QApplication
)
from PyQt5.QtCore import Qt, QModelIndex, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QKeySequence

from dcp_client.gui._custom_qt_helpers import IconProvider, CustomItemDelegate
from dcp_client.gui.napari_window import NapariWindow
from dcp_client.gui._my_widget import MyWidget
from dcp_client.gui._filesystem_wig import MyQFileSystemModel, SegmentationFilterProxyModel

from dcp_client.utils import settings
from dcp_client.gui.feature_extraction_window import ExtractFeaturesDialog

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dcp_client.app import Application


class WorkerThread(QThread):
    """
    Worker thread for displaying Pulse ProgressBar during model serving.

    """

    task_finished = pyqtSignal(tuple)
    progress_updated = pyqtSignal(int, int)  # current, total

    def __init__(
        self,
        app: Application,
        task: str = None,
        parent=None,
        skip_images=None,
    ):
        """
        Initialize the WorkerThread.

        :param app: The Application instance. See dcp_client.app for more information.
        :type app: dcp_client.app.Application
        :param task: The task performed by the worker thread. Can be 'inference' or 'train'.
        :type task: str, optional
        :param parent: The parent QObject (default is None).
        :param skip_images: Optional set/list of image names to skip during segmentation.
        :type skip_images: set or list, optional
        """
        super().__init__(parent)
        self.app = app
        self.task = task
        self.skip_images = skip_images

    def run(self) -> None:
        """
        Once run_inference or run_train is executed, the tuple of
        (message_text, message_title) will be returned to on_finished.
        """
        try:
            if self.task == "inference":
                message_text, message_title = self.app.run_inference(progress_callback=self.on_progress, skip_images=self.skip_images)
            elif self.task == "train":
                message_text, message_title = self.app.run_train()
            else:
                message_text, message_title = "Unknown task", "Error"

        except Exception as e:
            # Log any exceptions that might occur in the thread
            message_text, message_title = f"Exception in WorkerThread: {e}", "Error"

        self.task_finished.emit((message_text, message_title))

    def on_progress(self, current: int, total: int) -> None:
        """
        Callback to emit progress updates.

        :param current: Number of images processed so far.
        :type current: int
        :param total: Total number of images to process.
        :type total: int
        """
        self.progress_updated.emit(current, total)


class MainWindow(MyWidget):
    """
    Main Window Widget object.
    Opens the main window of the app where selected images in both directories are listed.
    User can view the images, train the model to get the labels, and visualise the result.

    """

    def __init__(self, app: Application) -> None:
        """
        Initializes the MainWindow.

        :param app: The Application instance. See dcp_client.app for more information.
        :type app: dcp_client.app.Application
        :param app.uncur_data_path: Chosen path to images without labels, selected by the user in the WelcomeWindow.
        :type app.uncur_data_path: str
        :param app.cur_data_path: Chosen path to images with labels, selected by the user in the WelcomeWindow.
        :type app.cur_data_path: str
        """

        super().__init__()
        self.app = app
        self.title = "DCP: Data Overview"
        self.worker_thread = None
        self.accepted_types = ['*'+end for end in settings.accepted_types]
        self.main_window()

    def main_window(self) -> None:
        """Sets up the GUI"""
        self.setWindowTitle(self.title)
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #f3f3f3;")

        main_layout = QVBoxLayout()
        dir_layout = QHBoxLayout()

        # create three boxes, for the three folder layouts
        self.uncurated_layout = QVBoxLayout()
        self.inprogress_layout = QVBoxLayout()
        self.curated_layout = QVBoxLayout()
        
        # fill first box - uncurated layout
        self.eval_dir_layout = QVBoxLayout()
        self.eval_dir_layout.setContentsMargins(0, 0, 0, 0)
        # add label
        self.label_eval = QLabel(self)
        self.label_eval.setText("Uncurated Dataset")
        self.label_eval.setMinimumHeight(50)
        self.label_eval.setMinimumWidth(200)
        self.label_eval.setAlignment(Qt.AlignCenter)
        self.label_eval.setStyleSheet(
            """
            font-size: 20px;
            font-weight: bold; 
            background-color: #015998;
            color: #ffffff;
            border-radius: 5px; 
            padding: 8px 16px;"""
        )
        self.eval_dir_layout.addWidget(self.label_eval)
        # add eval dir list
        model_eval = MyQFileSystemModel(app=self.app)
        model_eval.setRootPath("/")
        model_eval.setNameFilters(self.accepted_types)
        model_eval.setNameFilterDisables(False)  # Enable the filters
        model_eval.setIconProvider(IconProvider())
        model_eval.sort(0, Qt.AscendingOrder)
        
        # Wrap with proxy model to filter out _seg files
        proxy_eval = SegmentationFilterProxyModel()
        proxy_eval.setSourceModel(model_eval)
        proxy_eval.setFilterCaseSensitivity(Qt.CaseInsensitive)

        self.list_view_eval = QTreeView(self)
        self.list_view_eval.setToolTip("To visualize an image double click on it, or select it and then hit Enter")
        self.list_view_eval.setIconSize(QSize(128, 128))
        self.list_view_eval.setStyleSheet("background-color: #ffffff")
        self.list_view_eval.setModel(proxy_eval)
        self.list_view_eval.setItemDelegate(CustomItemDelegate())

        for i in range(1, 4):
            self.list_view_eval.hideColumn(i)
        
        # Set root index for the proxy model
        source_root_index = model_eval.setRootPath(self.app.uncur_data_path)
        proxy_root_index = proxy_eval.mapFromSource(source_root_index)
        self.list_view_eval.setRootIndex(proxy_root_index)
        
        self.list_view_eval.clicked.connect(self.on_item_eval_selected)
        self.list_view_eval.doubleClicked.connect(self.on_item_eval_double_clicked)

        self.list_view_eval.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.eval_dir_layout.addWidget(self.list_view_eval, 1)
        self.uncurated_layout.addLayout(self.eval_dir_layout)
        
        # Store reference to the model for cache clearing
        self.model_eval = model_eval

        # add run inference button
        self.inference_button = QPushButton("Generate Labels", self)
        self.inference_button.setStyleSheet(
            """QPushButton 
            { 
                  background-color: #3d81d1;
                  font-size: 12px; 
                  font-weight: bold;
                  color: #ffffff; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
            "QPushButton:hover { background-color: #7bc432; }"
            "QPushButton:pressed { background-color: #7bc432; }"
        )
        self.inference_button.clicked.connect(self.on_run_inference_button_clicked)
        # buttons will be placed in a full-width widget below the file lists

        dir_layout.addLayout(self.uncurated_layout)

        # In progress layout
        self.inprogr_dir_layout = QVBoxLayout()
        self.inprogr_dir_layout.setContentsMargins(0, 0, 0, 0)
        # Add in progress layout
        self.label_inprogr = QLabel(self)
        self.label_inprogr.setMinimumHeight(50)
        self.label_inprogr.setMinimumWidth(200)
        self.label_inprogr.setAlignment(Qt.AlignCenter)
        self.label_inprogr.setStyleSheet(
            "font-size: 20px; font-weight: bold; background-color: #015998; color: #ffffff; border-radius: 5px; padding: 8px 16px;"
        )
        self.label_inprogr.setText("Curation in progress")
        self.inprogr_dir_layout.addWidget(self.label_inprogr)
        # add in progress dir list
        model_inprogr = MyQFileSystemModel(app=self.app)
        model_inprogr.setRootPath("/")
        model_inprogr.setNameFilters(self.accepted_types)
        model_inprogr.setNameFilterDisables(False)  # Enable the filters
        model_inprogr.setIconProvider(IconProvider())
        
        # Wrap with proxy model to filter out _seg files
        proxy_inprogr = SegmentationFilterProxyModel()
        proxy_inprogr.setSourceModel(model_inprogr)
        proxy_inprogr.setFilterCaseSensitivity(Qt.CaseInsensitive)
        
        self.list_view_inprogr = QTreeView(self)
        self.list_view_inprogr.setToolTip("Select an image, click it, then press Enter")
        self.list_view_inprogr.setIconSize(QSize(128, 128))
        self.list_view_inprogr.setStyleSheet("background-color: #ffffff")
        self.list_view_inprogr.setModel(proxy_inprogr)
        self.list_view_inprogr.setItemDelegate(CustomItemDelegate())
        self.list_view_inprogr.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        for i in range(1, 4):
            self.list_view_inprogr.hideColumn(i)
        
        # Set root index for the proxy model
        source_root_index_inprogr = model_inprogr.setRootPath(self.app.inprogr_data_path)
        proxy_root_index_inprogr = proxy_inprogr.mapFromSource(source_root_index_inprogr)
        self.list_view_inprogr.setRootIndex(proxy_root_index_inprogr)
        
        self.list_view_inprogr.clicked.connect(self.on_item_inprogr_selected)
        self.list_view_inprogr.doubleClicked.connect(self.on_item_inprogr_double_clicked)
        self.inprogr_dir_layout.addWidget(self.list_view_inprogr, 1)
        self.inprogress_layout.addLayout(self.inprogr_dir_layout)
        
        # Store reference to the model for cache clearing
        self.model_inprogr = model_inprogr

        # the launch napari viewer button is currently hidden!
        launch_nap_button = QPushButton()
        launch_nap_button.setStyleSheet(
            "QPushButton { background-color: transparent; border: none; border-radius: 5px; padding: 8px 16px; }"
        )

        launch_nap_button.setEnabled(False)
        #self.inprogress_layout.addWidget(launch_nap_button, alignment=Qt.AlignCenter)
        dir_layout.addLayout(self.inprogress_layout)
        # Create a shortcut for the Enter key to click the button
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.on_launch_napari_button_clicked)

        # Curated layout
        self.train_dir_layout = QVBoxLayout()
        self.train_dir_layout.setContentsMargins(0, 0, 0, 0)
        self.label_train = QLabel(self)
        self.label_train.setText("Curated dataset")
        self.label_train.setMinimumHeight(50)
        self.label_train.setMinimumWidth(200)
        self.label_train.setAlignment(Qt.AlignCenter)
        self.label_train.setStyleSheet(
            "font-size: 20px; font-weight: bold; background-color: #015998; color: #ffffff; border-radius: 5px; padding: 8px 16px;"
        )
        self.train_dir_layout.addWidget(self.label_train)
        # add train dir list
        model_train = MyQFileSystemModel(app=self.app)
        model_train.setRootPath("/")
        model_train.setNameFilters(self.accepted_types)
        model_train.setNameFilterDisables(False)  # Enable the filters
        model_train.setIconProvider(IconProvider())
        
        # Wrap with proxy model to filter out _seg files
        proxy_train = SegmentationFilterProxyModel()
        proxy_train.setSourceModel(model_train)
        proxy_train.setFilterCaseSensitivity(Qt.CaseInsensitive)
        
        self.list_view_train = QTreeView(self)
        self.list_view_train.setToolTip("Select an image, click it, then press Enter")
        self.list_view_train.setIconSize(QSize(128, 128))
        self.list_view_train.setStyleSheet("background-color: #ffffff")
        self.list_view_train.setModel(proxy_train)
        self.list_view_train.setItemDelegate(CustomItemDelegate())

        for i in range(1, 4):
            self.list_view_train.hideColumn(i)
        
        # Set root index for the proxy model
        source_root_index_train = model_train.setRootPath(self.app.cur_data_path)
        proxy_root_index_train = proxy_train.mapFromSource(source_root_index_train)
        self.list_view_train.setRootIndex(proxy_root_index_train)
        
        self.list_view_train.clicked.connect(self.on_item_train_selected)
        self.list_view_train.doubleClicked.connect(self.on_item_train_double_clicked)
        self.list_view_train.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.train_dir_layout.addWidget(self.list_view_train, 1)
        self.curated_layout.addLayout(self.train_dir_layout)

        # add extract features button under curated dataset
        self.extract_button = QPushButton("Extract features", self)
        self.extract_button.setStyleSheet(
            """QPushButton 
            { 
                  background-color: #3d81d1;
                  font-size: 12px; 
                  font-weight: bold;
                  color: #ffffff; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
            "QPushButton:hover { background-color: #7bc432; }"
            "QPushButton:pressed { background-color: #7bc432; }"
        )
        self.extract_button.clicked.connect(self.on_extract_features_clicked)
        # extract button moved to bottom button bar
        
        # Store reference to the model for cache clearing
        self.model_train = model_train
        '''
        # add train button
        self.train_button = QPushButton("Train Model", self)
        self.train_button.setStyleSheet(
            """QPushButton 
            { 
                  background-color: #3d81d1;
                  font-size: 12px; 
                  font-weight: bold;
                  color: #ffffff; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
            "QPushButton:hover { background-color: #7bc432; }"
            "QPushButton:pressed { background-color: #7bc432; }"
        )
        self.train_button.clicked.connect(self.on_train_button_clicked)
        self.curated_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)
        '''
        dir_layout.addLayout(self.curated_layout)


        main_layout.addLayout(dir_layout)

        # Full-width button bar below the file-list columns with three containers
        buttons_widget = QWidget()
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 8, 0, 8)

        # Left container - holds Generate Labels under Uncurated column
        left_container = QWidget()
        left_layout = QHBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addStretch(1)
        left_layout.addWidget(self.inference_button, 0, Qt.AlignHCenter)
        left_layout.addStretch(1)
        left_container.setLayout(left_layout)
        left_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Middle container - empty (keeps spacing)
        mid_container = QWidget()
        mid_layout = QHBoxLayout()
        mid_layout.setContentsMargins(0, 0, 0, 0)
        mid_layout.addStretch(1)
        mid_container.setLayout(mid_layout)
        mid_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Right container - holds Extract features under Curated column
        right_container = QWidget()
        right_layout = QHBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addStretch(1)
        right_layout.addWidget(self.extract_button, 0, Qt.AlignHCenter)
        right_layout.addStretch(1)
        right_container.setLayout(right_layout)
        right_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        btn_layout.addWidget(left_container)
        btn_layout.addWidget(mid_container)
        btn_layout.addWidget(right_container)

        buttons_widget.setLayout(btn_layout)
        buttons_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        main_layout.addWidget(buttons_widget)

        # add progress bar with hint text
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimumWidth(700)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addStretch(1)

        self.hint_label = QLabel("Double click on an image to launch the viewer!")
        self.hint_label.setStyleSheet("color: #666666; font-size: 11px; font-style: italic;")
        self.hint_label.setContentsMargins(15, 0, 10, 0)
        self.hint_label.setVisible(True)
        progress_layout.addWidget(self.hint_label)
        
        main_layout.addLayout(progress_layout)

        # add it all to main layout and show
        self.setLayout(main_layout)
        self.show()

    def on_item_train_selected(self, item: QModelIndex) -> None:
        """
        Is called once an image is selected in the 'curated dataset' folder.

        :param item: The selected item from the 'curated dataset' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.cur_data_path

    def on_item_eval_selected(self, item: QModelIndex) -> None:
        """
        Is called once an image is selected in the 'uncurated dataset' folder.

        :param item: The selected item from the 'uncurated dataset' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.uncur_data_path

    def on_item_inprogr_selected(self, item: QModelIndex) -> None:
        """
        Is called once an image is selected in the 'in progress' folder.

        :param item: The selected item from the 'in progress' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.inprogr_data_path

    def on_item_eval_double_clicked(self, item: QModelIndex) -> None:
        """
        Is called once an image is double-clicked in the 'uncurated dataset' folder.
        Launches the napari viewer immediately.

        :param item: The selected item from the 'uncurated dataset' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.uncur_data_path
        self.on_launch_napari_button_clicked()

    def on_item_inprogr_double_clicked(self, item: QModelIndex) -> None:
        """
        Is called once an image is double-clicked in the 'in progress' folder.
        Launches the napari viewer immediately.

        :param item: The selected item from the 'in progress' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.inprogr_data_path
        self.on_launch_napari_button_clicked()

    def on_item_train_double_clicked(self, item: QModelIndex) -> None:
        """
        Is called once an image is double-clicked in the 'curated dataset' folder.
        Launches the napari viewer immediately.

        :param item: The selected item from the 'curated dataset' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.cur_data_path
        self.on_launch_napari_button_clicked()
    '''
    def on_train_button_clicked(self) -> None:
        """
        Is called once user clicks the "Train Model" button.
        """
        self.train_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        # initialise the worker thread
        self.worker_thread = WorkerThread(app=self.app, task="train")
        self.worker_thread.task_finished.connect(self.on_finished)
        # start the worker thread to train
        self.worker_thread.start()
    '''
    def on_run_inference_button_clicked(self) -> None:
        """
        Is called once user clicks the "Generate Labels" button.
        """
        # Check for existing segmentations
        existing_segs = self.app.check_existing_segmentations()
        
        skip_images = set()
        if existing_segs:
            # Existing segmentations found - ask user what to do
            user_choice = self.create_segmentation_option_dialog(num_images_with_segs=len(existing_segs))
            
            if user_choice == "skip":
                # Skip images with existing segmentations
                skip_images = set(existing_segs.keys())
            elif user_choice == "cancel":
                # User cancelled the operation
                return
            # If "regenerate", skip_images remains empty and all images will be processed
        
        self.inference_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 1)  # Default range; will be updated by progress
        # initialise the worker thread
        self.worker_thread = WorkerThread(app=self.app, task="inference", skip_images=skip_images)
        self.worker_thread.task_finished.connect(self.on_finished)
        self.worker_thread.progress_updated.connect(self.on_progress_updated)
        # start the worker thread to run inference
        self.worker_thread.start()

    def on_progress_updated(self, current: int, total: int) -> None:
        """
        Updates the progress bar based on the number of images segmented.
        
        :param current: Number of images processed so far.
        :type current: int
        :param total: Total number of images to process.
        :type total: int
        """
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def on_extract_features_clicked(self) -> None:
        """Opens the extract features dialog."""
        dlg = ExtractFeaturesDialog(parent=self, default_filename="extracted_features.csv")
        dlg.exec_()

    def clear_image_caches(self) -> None:
        """
        Clears the image caches from all file system models.
        This should be called after new segmentations are created to force
        the views to reload and overlay the new segmentations.
        """
        if hasattr(self, 'model_eval'):
            self.model_eval.clear_cache()
        if hasattr(self, 'model_inprogr'):
            self.model_inprogr.clear_cache()
        if hasattr(self, 'model_train'):
            self.model_train.clear_cache()

    def on_launch_napari_button_clicked(self):
        ''' 
        Launches the napari window after the image is selected.
        '''
        if not self.app.cur_selected_img or '_seg.tiff' in self.app.cur_selected_img:
            message_text = "Please first select an image you wish to visualize. The selected image must be an original image, not a mask."
            _ = self.create_warning_box(message_text, message_title="Warning")
        else:
            try:
                self.nap_win = NapariWindow(self.app)
                self.nap_win.show() 
            except Exception as e:
                message_text = f"An error occurred while opening the Napari window: {str(e)}"
                _ = self.create_warning_box(message_text, message_title="Error")

    def on_finished(self, result: tuple) -> None:
        """
        Is called once the worker thread emits the on finished signal.

        :param result: The result emitted by the worker thread. See return type of WorkerThread.run
        :type result: tuple
        """
        logger.info("on_finished called")
        try:
            # Stop the pulsation
            logger.debug("Setting progress bar range to 0, 1")
            self.progress_bar.setRange(0, 1)
            
            # Display message of result (compute message first)
            message_text, message_title = result
            logger.info(f"Task completed - Title: {message_title}, Message: {message_text[:100]}...")

            # Clean up worker thread safely before showing a blocking dialog.
            # Showing a modal dialog (QMessageBox.exec) while background threads
            # are still shutting down can lead to deadlocks on some systems
            # when logging streams are written from those threads. Moving cleanup
            # first avoids that class of issues.
            if self.worker_thread is not None:
                logger.debug("Disconnecting worker thread signals")
                try:
                    self.worker_thread.task_finished.disconnect(self.on_finished)
                except RuntimeError as e:
                    logger.warning(f"Signal disconnect failed: {e}")

                logger.debug("Quitting worker thread")
                self.worker_thread.quit()

                logger.debug("Waiting for worker thread to finish (timeout: 5 seconds)")
                if not self.worker_thread.wait(5000):  # 5 second timeout
                    logger.warning("Worker thread did not finish within timeout, forcing termination")
                    self.worker_thread.terminate()
                    self.worker_thread.wait()
                else:
                    logger.debug("Worker thread finished successfully")

                logger.debug("Scheduling worker thread deletion")
                self.worker_thread.deleteLater()
                self.worker_thread = None  # Set to None to indicate it's no longer in use
                logger.info("Worker thread cleanup completed")
            else:
                logger.warning("Worker thread was already None")

            # Now show the (blocking) message box to the user and re-enable buttons.
            logger.debug("Creating warning box")
            _ = self.create_warning_box(message_text, message_title)
            logger.debug("Warning box closed")

            # Clear image caches to reload images with new segmentations
            logger.debug("Clearing image caches")
            self.clear_image_caches()

            # Re-enable buttons
            logger.debug("Re-enabling inference button")
            self.inference_button.setEnabled(True)
                
        except Exception as e:
            logger.error(f"Error in on_finished: {e}", exc_info=True)
            self.inference_button.setEnabled(True)


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from dcp_client.app import Application
    from dcp_client.utils.bentoml_model import BentomlModel
    from dcp_client.utils.fsimagestorage import FilesystemImageStorage
    from dcp_client.utils import settings

    settings.init()
    image_storage = FilesystemImageStorage()
    ml_model = BentomlModel()
    app = QApplication(sys.argv)
    app_ = Application(
        ml_model=ml_model,
        num_classes=1,
        image_storage=image_storage,
        server_ip="0.0.0.0",
        server_port=7010,
        uncur_data_path="data",
        cur_data_path="",  # set path
        inprogr_data_path="",
    )  # set path
    window = MainWindow(app=app_)
    sys.exit(app.exec())
