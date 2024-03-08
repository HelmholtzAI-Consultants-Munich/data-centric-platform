from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QFileSystemModel, QHBoxLayout, QLabel, QTreeView, QProgressBar, QShortcut
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRect, QModelIndex
from PyQt5.QtGui import QKeySequence

from PyQt5.QtCore import  QVariant, QDir
from PyQt5.QtGui import QPixmap, QPainter, QImage, QBrush
from PyQt5.QtWidgets import QApplication, QStyledItemDelegate

from dcp_client.utils import settings
from dcp_client.utils.utils import IconProvider, CustomItemDelegate

from dcp_client.gui.napari_window import NapariWindow
from dcp_client.gui._my_widget import MyWidget

if TYPE_CHECKING:
    from dcp_client.app import Application


class WorkerThread(QThread):
    ''' Worker thread for displaying Pulse ProgressBar during model serving '''
    task_finished = pyqtSignal(tuple)
    def __init__(self, app: Application, task: str = None, parent = None,):
        super().__init__(parent)
        self.app = app
        self.task = task

    def run(self):
        ''' Once run_inference the tuple of (message_text, message_title) will be returned to on_finished'''
        try:
            if self.task == 'inference':
                message_text, message_title = self.app.run_inference()
            elif self.task == 'train':
                message_text, message_title = self.app.run_train()
            else:
                message_text, message_title = "Unknown task", "Error"

        except Exception as e:
            # Log any exceptions that might occur in the thread
            message_text, message_title = f"Exception in WorkerThread: {e}", "Error"

        self.task_finished.emit((message_text, message_title))



class MyQFileSystemModel(QFileSystemModel):
    def __init__(self):
        """
        Initializes a custom QFileSystemModel
        """
        super().__init__()
    
    # def sort(self, column, order):
    #     super().sort(column, order)
    #     if order == Qt.AscendingOrder:
    #         fileList = [self.data(self.index(row, column)) for row in range(self.rowCount())]
    #         fileList.sort(key=lambda path: '_seg' in path)
    #     elif order == Qt.DescendingOrder:
    #         fileList = [self.data(self.index(row, column)) for row in range(self.rowCount())]
    #         fileList.sort(key=lambda path: '_seg' not in path)
    #     self.setRootPath(self.rootPath())  

    def setFilter(self, filters):
        """
        Sets filters for the model.

        :param filters: The filters to be applied. (QDir.Filters)
        """
        filters |= QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files
        super().setFilter(filters)

        # Exclude files containing '_seg' in their names
        self.addFilter(lambda fileInfo: "_seg" not in fileInfo.fileName())

    def addFilter(self, filterFunc):
        """
        Adds a custom filter function to the model.

        :param filterFunc: The filter function to be added. (function)
        """
        self.filterFunc = filterFunc

    def headerData(self, section, orientation, role):
        """
        Reimplemented method to provide custom header data for the model's headers.

        :param section: The section (column) index. (int)
        :param orientation: The orientation of the header. (Qt.Orientation)
        :param role: The role of the header data. (int)
        :rtype: QVariant
        """
        if section == 0 and role == Qt.DisplayRole:
            return ""  
        else:
            return super().headerData(section, orientation, role)

    def data(self, index, role=Qt.DisplayRole):
        """
        Reimplemented method to provide custom data for the model's items.

        :param index: The index of the item. (QModelIndex)
        :param role: The role of the data. (int)
        :rtype: QVariant
        """
        if not index.isValid():
            return QVariant()

        if role == Qt.DisplayRole:
            filepath_img = self.filePath(index)
            if '_seg' in filepath_img:
                return None

        if role == Qt.DecorationRole:
            filepath_img = self.filePath(index)
            if filepath_img.endswith('.tiff') or filepath_img.endswith('.png'):
                if '_seg' not in filepath_img:
                    painter = QPainter()

                    filepath_mask = f"{filepath_img.split('.')[0]}_seg.tiff"
                    img = QImage(filepath_img).scaled(96, 96)

                    if os.path.exists(filepath_mask):
                        painter.begin(img)

                        rect_corner_left = QRect(0, 0, 32, 32)
                        rect_corner_bot = QRect(64, 64, 32, 32)

                        rect_left = QRect(32, 0, 64, 32)
                        rect_bot = QRect(64, 0, 32, 64)

                        painter.fillRect(rect_corner_left, QBrush(Qt.white, Qt.SolidPattern))
                        painter.fillRect(rect_corner_bot, QBrush(Qt.white, Qt.SolidPattern))

                        painter.fillRect(rect_left, QBrush(Qt.black, Qt.SolidPattern))
                        painter.fillRect(rect_bot, QBrush(Qt.black, Qt.SolidPattern))

                        painter.end()

                        pixmap = QPixmap.fromImage(img)

                        return pixmap

                    else:
                        return img

                else:
                    return None

        return super().data(index, role)


class ImageDelegate(QStyledItemDelegate):
    """
    Custom delegate for displaying images in Qt views.
    """

    def paint(self, painter, option, index):
        """
        Reimplemented method to paint the item.

        :param painter: The QPainter used for painting. (QPainter)
        :param option: The style options for the item. (QStyleOptionViewItem)
        :param index: The model index of the item. (QModelIndex)
        """
        if index.isValid() and index.data(Qt.DecorationRole):
            pixmap = index.data(Qt.DecorationRole)
            painter.drawPixmap(option.rect, pixmap)



class MainWindow(MyWidget):
    '''
    Main Window Widget object.
    Opens the main window of the app where selected images in both directories are listed. 
    User can view the images, train the mdoel to get the labels, and visualise the result.
    :param eval_data_path: Chosen path to images without labeles, selected by the user in the WelcomeWindow
    :type eval_data_path: string
    :param train_data_path: Chosen path to images with labeles, selected by the user in the WelcomeWindow
    :type train_data_path: string
    '''    


    def __init__(self, app: Application):
        super().__init__()
        self.app = app
        self.title = "Data Overview"
        self.worker_thread = None
        self.main_window()
        
    def main_window(self):
        '''
        Sets up the GUI
        '''
        self.setWindowTitle(self.title)
        self.resize(1000, 700)
        self.setStyleSheet("background-color: #f3f3f3;")
        main_layout = QVBoxLayout()
        dir_layout = QHBoxLayout()  
        
        self.uncurated_layout = QVBoxLayout()
        self.inprogress_layout = QVBoxLayout()
        self.curated_layout = QVBoxLayout()
        
        self.eval_dir_layout = QVBoxLayout() 
        self.eval_dir_layout.setContentsMargins(0,0,0,0)

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
       
        model_eval = MyQFileSystemModel()
        model_eval.setIconProvider(IconProvider())
        model_eval.sort(0, Qt.AscendingOrder)
 
      
        self.list_view_eval = QTreeView(self)

        self.list_view_eval.setItemDelegate(ImageDelegate())

        self.list_view_eval.setToolTip("Select an image, click it, then press Enter")
        # self.list_view_eval.setIconSize(QSize(50,50))
        self.list_view_eval.setStyleSheet("background-color: #ffffff")
        self.list_view_eval.setModel(model_eval)
        model_eval.setRootPath('/')
        delegate = CustomItemDelegate()
        self.list_view_eval.setItemDelegate(delegate)

        for i in range(1,4):
            self.list_view_eval.hideColumn(i)
        #self.list_view_eval.setFixedSize(600, 600)
        self.list_view_eval.setRootIndex(model_eval.setRootPath(self.app.eval_data_path)) 
        self.list_view_eval.clicked.connect(self.on_item_eval_selected)
        
        self.eval_dir_layout.addWidget(self.list_view_eval)
        self.uncurated_layout.addLayout(self.eval_dir_layout)

        # add buttons
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
        self.inference_button.clicked.connect(self.on_run_inference_button_clicked)  # add selected image    
        self.uncurated_layout.addWidget(self.inference_button, alignment=Qt.AlignCenter)

        dir_layout.addLayout(self.uncurated_layout)

        # In progress layout
        self.inprogr_dir_layout = QVBoxLayout() 
        self.inprogr_dir_layout.setContentsMargins(0,0,0,0)

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
        model_inprogr = MyQFileSystemModel()
      
        #self.list_view = QListView(self)
        self.list_view_inprogr = QTreeView(self)
        # self.list_view_inprogr.setIconSize(QSize(50,50))
        self.list_view_inprogr.setStyleSheet("background-color: #ffffff")
        model_inprogr.setIconProvider(IconProvider())
        self.list_view_inprogr.setModel(model_inprogr)

        model_inprogr.setRootPath('/')
        delegate = CustomItemDelegate()
        self.list_view_inprogr.setItemDelegate(delegate)

        for i in range(1,4):
            self.list_view_inprogr.hideColumn(i)
        #self.list_view_inprogr.setFixedSize(600, 600)
        self.list_view_inprogr.setRootIndex(model_inprogr.setRootPath(self.app.inprogr_data_path)) 
        self.list_view_inprogr.clicked.connect(self.on_item_inprogr_selected)
        self.inprogr_dir_layout.addWidget(self.list_view_inprogr)
        self.inprogress_layout.addLayout(self.inprogr_dir_layout)

        self.launch_nap_button = QPushButton()
        self.launch_nap_button.setStyleSheet(
        "QPushButton { background-color: transparent; border: none; border-radius: 5px; padding: 8px 16px; }"
        )
        

        self.launch_nap_button.setEnabled(False)
        self.inprogress_layout.addWidget(self.launch_nap_button, alignment=Qt.AlignCenter)

        # Create a shortcut for the Enter key to click the button
        enter_shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        enter_shortcut.activated.connect(self.on_launch_napari_button_clicked)

        dir_layout.addLayout(self.inprogress_layout)

        # Curated layout
        self.train_dir_layout = QVBoxLayout() 
        self.train_dir_layout.setContentsMargins(0,0,0,0)
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
        model_train = MyQFileSystemModel()
        # model_train.setNameFilters(["*_seg.tiff"])
        #self.list_view = QListView(self)
        self.list_view_train = QTreeView(self)
        # self.list_view_train.setIconSize(QSize(50,50))
        self.list_view_train.setStyleSheet("background-color: #ffffff")
        model_train.setIconProvider(IconProvider())
        self.list_view_train.setModel(model_train)

        model_train.setRootPath('/')
        delegate = CustomItemDelegate()
        self.list_view_train.setItemDelegate(delegate)

        for i in range(1,4):
            self.list_view_train.hideColumn(i)
        #self.list_view_train.setFixedSize(600, 600)
        self.list_view_train.setRootIndex(model_train.setRootPath(self.app.train_data_path)) 
        self.list_view_train.clicked.connect(self.on_item_train_selected)
        self.train_dir_layout.addWidget(self.list_view_train)
        self.curated_layout.addLayout(self.train_dir_layout)
        
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
        self.train_button.clicked.connect(self.on_train_button_clicked)  # add selected image    
        self.curated_layout.addWidget(self.train_button, alignment=Qt.AlignCenter)
        dir_layout.addLayout(self.curated_layout)

        main_layout.addLayout(dir_layout)

        # add progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addStretch(1) 
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimumWidth(1000)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setRange(0,1)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addLayout(progress_layout)

        self.setLayout(main_layout)
        self.show()

    def on_item_train_selected(self, item):
        '''
        Is called once an image is selected in the 'curated dataset' folder
        '''
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.train_data_path

    def on_item_eval_selected(self, item):
        '''
        Is called once an image is selected in the 'uncurated dataset' folder
        '''
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.eval_data_path

    def on_item_inprogr_selected(self, item):
        '''
        Is called once an image is selected in the 'in progress' folder
        '''
        self.app.cur_selected_img = item.data()
        self.app.cur_selected_path = self.app.inprogr_data_path

    def on_train_button_clicked(self):
        ''' 
        Is called once user clicks the "Train Model" button
        '''
        self.train_button.setEnabled(False)
        self.progress_bar.setRange(0,0)
        # initialise the worker thread
        self.worker_thread = WorkerThread(app=self.app, task='train')
        self.worker_thread.task_finished.connect(self.on_finished)
        # start the worker thread to train
        self.worker_thread.start()

    def on_run_inference_button_clicked(self):
        ''' 
        Is called once user clicks the "Generate Labels" button
        '''
        self.inference_button.setEnabled(False)
        self.progress_bar.setRange(0,0)
        # initialise the worker thread
        self.worker_thread = WorkerThread(app=self.app, task='inference')
        self.worker_thread.task_finished.connect(self.on_finished)
        # start the worker thread to run inference
        self.worker_thread.start()

    def on_launch_napari_button_clicked(self):   
        ''' 
        Launches the napari window after the image is selected.
        '''
        if not self.app.cur_selected_img or '_seg.tiff' in self.app.cur_selected_img:
            message_text = "Please first select an image you wish to visualise. The selected image must be an original image, not a mask."
            _ = self.create_warning_box(message_text, message_title="Warning")
        else:
            self.nap_win = NapariWindow(self.app)
            self.nap_win.show() 

    def on_finished(self, result):
        ''' 
        Is called once the worker thread emits the on finished signal
        '''
        # Stop the pulsation
        self.progress_bar.setRange(0,1) 
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
    data_sync = DataRSync(user_name="local",
                          host_name="local",
                          server_repo_path=None)
    app = QApplication(sys.argv)
    app_ = Application(ml_model=ml_model, 
                       syncer=data_sync,
                       image_storage=image_storage, 
                       server_ip='0.0.0.0',
                       server_port=7010,
                       eval_data_path='data', 
                       train_data_path='', # set path
                       inprogr_data_path='') # set path
    window = MainWindow(app=app_)
    sys.exit(app.exec())