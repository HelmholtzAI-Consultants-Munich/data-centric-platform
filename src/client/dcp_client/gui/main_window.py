from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import (
    QPushButton,
    QVBoxLayout,
    QFileSystemModel,
    QHBoxLayout,
    QLabel,
    QTreeView,
    QProgressBar,
    QShortcut,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence

from dcp_client.utils import settings
from dcp_client.utils.utils import IconProvider

from dcp_client.gui.napari_window import NapariWindow
from dcp_client.gui._my_widget import MyWidget

if TYPE_CHECKING:
    from dcp_client.app import Application


class WorkerThread(QThread):
    """
    Worker thread for displaying Pulse ProgressBar during model serving.

    """

    task_finished = pyqtSignal(tuple)

    def __init__(
        self,
        app: Application,
        task: str = None,
        parent=None,
    ):
        """
        Initialize the WorkerThread.

        :param app: The Application instance. See dcp_client.app for more information.
        :type app: dcp_client.app.Application
        :param task: The task performed by the worker thread. Can be 'inference' or 'train'.
        :type task: str, optional
        :param parent: The parent QObject (default is None).
        """
        super().__init__(parent)
        self.app = app
        self.task = task

    def run(self):
        """
        Once run_inference or run_train is executed, the tuple of
        (message_text, message_title) will be returned to on_finished.
        """
        try:
            if self.task == "inference":
                message_text, message_title = self.app.run_inference()
            elif self.task == "train":
                message_text, message_title = self.app.run_train()
            else:
                message_text, message_title = "Unknown task", "Error"

        except Exception as e:
            # Log any exceptions that might occur in the thread
            message_text, message_title = f"Exception in WorkerThread: {e}", "Error"

        self.task_finished.emit((message_text, message_title))


class MainWindow(MyWidget):
    """
    Main Window Widget object.
    Opens the main window of the app where selected images in both directories are listed.
    User can view the images, train the model to get the labels, and visualise the result.

    :param eval_data_path: Chosen path to images without labeles, selected by the user in the WelcomeWindow
    :type eval_data_path: string
    :param train_data_path: Chosen path to images with labeles, selected by the user in the WelcomeWindow
    :type train_data_path: string
    """

    def __init__(self, app: Application):
        """
        Initializes the MainWindow.

        :param app: The Application instance. See dcp_client.app for more information.
        :type app: dcp_client.app.Application
        :param app.eval_data_path: Chosen path to images without labels, selected by the user in the WelcomeWindow.
        :type app.eval_data_path: str
        :param app.train_data_path: Chosen path to images with labels, selected by the user in the WelcomeWindow.
        :type app.train_data_path: str
        """
        super().__init__()
        self.app = app
        self.title = "Data Overview"
        self.worker_thread = None
        self.main_window()

    def main_window(self):
        """Sets up the GUI"""
        self.setWindowTitle(self.title)
        # self.resize(1000, 1500)
        main_layout = QVBoxLayout()
        dir_layout = QHBoxLayout()

        self.uncurated_layout = QVBoxLayout()
        self.inprogress_layout = QVBoxLayout()
        self.curated_layout = QVBoxLayout()

        self.eval_dir_layout = QVBoxLayout()
        self.eval_dir_layout.setContentsMargins(0, 0, 0, 0)
        self.label_eval = QLabel(self)
        self.label_eval.setText("Uncurated dataset")
        self.eval_dir_layout.addWidget(self.label_eval)
        # add eval dir list
        model_eval = QFileSystemModel()
        model_eval.setIconProvider(IconProvider())
        self.list_view_eval = QTreeView(self)
        self.list_view_eval.setModel(model_eval)
        for i in range(1, 4):
            self.list_view_eval.hideColumn(i)
        # self.list_view_eval.setFixedSize(600, 600)
        self.list_view_eval.setRootIndex(
            model_eval.setRootPath(self.app.eval_data_path)
        )
        self.list_view_eval.clicked.connect(self.on_item_eval_selected)

        self.eval_dir_layout.addWidget(self.list_view_eval)
        self.uncurated_layout.addLayout(self.eval_dir_layout)

        # add buttons
        self.inference_button = QPushButton("Generate Labels", self)
        self.inference_button.clicked.connect(
            self.on_run_inference_button_clicked
        )  # add selected image
        self.uncurated_layout.addWidget(self.inference_button, alignment=Qt.AlignCenter)

        dir_layout.addLayout(self.uncurated_layout)

        # In progress layout
        self.inprogr_dir_layout = QVBoxLayout()
        self.inprogr_dir_layout.setContentsMargins(0, 0, 0, 0)
        self.label_inprogr = QLabel(self)
        self.label_inprogr.setText("Curation in progress")
        self.inprogr_dir_layout.addWidget(self.label_inprogr)
        # add in progress dir list
        model_inprogr = QFileSystemModel()
        # self.list_view = QListView(self)
        self.list_view_inprogr = QTreeView(self)
        model_inprogr.setIconProvider(IconProvider())
        self.list_view_inprogr.setModel(model_inprogr)
        for i in range(1, 4):
            self.list_view_inprogr.hideColumn(i)
        # self.list_view_inprogr.setFixedSize(600, 600)
        self.list_view_inprogr.setRootIndex(
            model_inprogr.setRootPath(self.app.inprogr_data_path)
        )
        self.list_view_inprogr.clicked.connect(self.on_item_inprogr_selected)
        self.inprogr_dir_layout.addWidget(self.list_view_inprogr)
        self.inprogress_layout.addLayout(self.inprogr_dir_layout)

        self.launch_nap_button = QPushButton("View image and fix label", self)
        self.launch_nap_button.clicked.connect(
            self.on_launch_napari_button_clicked
        )  # add selected image
        self.inprogress_layout.addWidget(
            self.launch_nap_button, alignment=Qt.AlignCenter
        )
        # Create a shortcut for the Enter key to click the button
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.on_launch_napari_button_clicked)

        dir_layout.addLayout(self.inprogress_layout)

        # Curated layout
        self.train_dir_layout = QVBoxLayout()
        self.train_dir_layout.setContentsMargins(0, 0, 0, 0)
        self.label_train = QLabel(self)
        self.label_train.setText("Curated dataset")
        self.train_dir_layout.addWidget(self.label_train)
        # add train dir list
        model_train = QFileSystemModel()
        # self.list_view = QListView(self)
        self.list_view_train = QTreeView(self)
        model_train.setIconProvider(IconProvider())
        self.list_view_train.setModel(model_train)
        for i in range(1, 4):
            self.list_view_train.hideColumn(i)
        # self.list_view_train.setFixedSize(600, 600)
        self.list_view_train.setRootIndex(
            model_train.setRootPath(self.app.train_data_path)
        )
        self.list_view_train.clicked.connect(self.on_item_train_selected)
        self.train_dir_layout.addWidget(self.list_view_train)
        self.curated_layout.addLayout(self.train_dir_layout)

        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(
            self.on_train_button_clicked
        )  # add selected image
        self.curated_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)
        dir_layout.addLayout(self.curated_layout)

        main_layout.addLayout(dir_layout)

        # add progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addStretch(1)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 1)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        self.setLayout(main_layout)
        self.show()

    def on_item_train_selected(self, item):
        """
        Is called once an image is selected in the 'curated dataset' folder.

        :param item: The selected item from the 'curated dataset' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.train_data_path

    def on_item_eval_selected(self, item):
        """
        Is called once an image is selected in the 'uncurated dataset' folder.

        :param item: The selected item from the 'uncurated dataset' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.eval_data_path

    def on_item_inprogr_selected(self, item):
        """
        Is called once an image is selected in the 'in progress' folder.

        :param item: The selected item from the 'in progress' folder.
        :type item: QModelIndex
        """
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.inprogr_data_path

    def on_train_button_clicked(self):
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

    def on_run_inference_button_clicked(self):
        """
        Is called once user clicks the "Generate Labels" button.
        """
        self.inference_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        # initialise the worker thread
        self.worker_thread = WorkerThread(app=self.app, task="inference")
        self.worker_thread.task_finished.connect(self.on_finished)
        # start the worker thread to run inference
        self.worker_thread.start()

    def on_launch_napari_button_clicked(self):
        """
        Launches the napari window after the image is selected.
        """
        if not self.app.cur_selected_img or "_seg.tiff" in self.app.cur_selected_img:
            message_text = "Please first select an image you wish to visualise. The selected image must be an original image, not a mask."
            _ = self.create_warning_box(message_text, message_title="Warning")
        else:
            self.nap_win = NapariWindow(self.app)
            self.nap_win.show()

    def on_finished(self, result):
        """
        Is called once the worker thread emits the on finished signal.

        :param result: The result emitted by the worker thread. See return type of WorkerThread.run
        :type result: tuple
        """
        # Stop the pulsation
        self.progress_bar.setRange(0, 1)
        # Display message of result
        message_text, message_title = result
        _ = self.create_warning_box(message_text, message_title)
        # Re-enable buttons
        self.inference_button.setEnabled(True)
        self.train_button.setEnabled(True)
        # Delete the worker thread when it's done
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.worker_thread.deleteLater()
        self.worker_thread = None  # Set to None to indicate it's no longer in use


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    from dcp_client.app import Application
    from dcp_client.utils.bentoml_model import BentomlModel
    from dcp_client.utils.fsimagestorage import FilesystemImageStorage
    from dcp_client.utils import settings
    from dcp_client.utils.sync_src_dst import DataRSync

    settings.init()
    image_storage = FilesystemImageStorage()
    ml_model = BentomlModel()
    data_sync = DataRSync(user_name="local", host_name="local", server_repo_path=None)
    app = QApplication(sys.argv)
    app_ = Application(
        ml_model=ml_model,
        syncer=data_sync,
        image_storage=image_storage,
        server_ip="0.0.0.0",
        server_port=7010,
        eval_data_path="data",
        train_data_path="",  # set path
        inprogr_data_path="",
    )  # set path
    window = MainWindow(app=app_)
    sys.exit(app.exec())
