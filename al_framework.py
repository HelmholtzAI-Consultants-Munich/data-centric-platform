import sys
import os
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMainWindow, QFileSystemModel, QListView, QHBoxLayout
from PyQt5.QtCore import QDir
import napari
from skimage.io import imread, imsave


def changeWindow(w1, w2):
    w1.hide()
    w2.show()

class NapariWindow(QWidget):
    def __init__(self, 
                img_filename,
                eval_data_path,
                train_data_path):
        super().__init__()
        self.img_filename = img_filename
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.img = imread(os.path.join(self.eval_data_path, self.img_filename))
        self.setWindowTitle("napari Viewer")
        self.viewer = napari.Viewer()
        self.viewer.add_image(self.img)
        main_window = self.viewer.window._qt_window
        layout = QVBoxLayout()
        layout.addWidget(main_window)
        self.return_button = QPushButton('Return')
        layout.addWidget(self.return_button)
        add_button = QPushButton('Add to training data')
        layout.addWidget(add_button)

        def on_add_button_clicked():
            seg = self.viewer.layers['Labels'].data
            os.replace(os.path.join(self.eval_data_path, self.img_filename), os.path.join(self.train_data_path, self.img_filename))
            seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
            imsave(os.path.join(self.train_data_path, seg_name),seg)

        def on_return_button_clicked():
            self.close()

        add_button.clicked.connect(on_add_button_clicked)
        self.return_button.clicked.connect(on_return_button_clicked)
        self.setLayout(layout)
        self.show()

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.title = "Data Overview"
        self.resize(1000, 1500)
        self.main_layout = QHBoxLayout()
        
        self.left_layout = QVBoxLayout()
        model_eval = QFileSystemModel()
        self.eval_data_path =  "/Users/christina.bukas/Desktop/eg_crops_450/"
        self.list_view = QListView(self)
        self.list_view.setModel(model_eval)
        self.list_view.setFixedSize(600, 600)
        self.list_view.setRootIndex(model_eval.setRootPath(self.eval_data_path)) 
        self.list_view.clicked.connect(self.item_eval_selected)
        self.cur_eval_img = None
        self.left_layout.addWidget(self.list_view)
        self.launch_nap_button = QPushButton("Annotate image", self)
        self.launch_nap_button.clicked.connect(self.launch_napari_window)  # add selected image    
        self.left_layout.addWidget(self.launch_nap_button)
        self.main_layout.addLayout(self.left_layout)


        self.right_layout = QVBoxLayout()    
        model_train = QFileSystemModel()
        self.train_data_path = "/Users/christina.bukas/Desktop/train/"
        self.list_view = QListView(self)
        self.list_view.setModel(model_train)
        self.list_view.setFixedSize(600, 600)
        self.list_view.setRootIndex(model_train.setRootPath(self.train_data_path)) 
        self.right_layout.addWidget(self.list_view)
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)  # add selected image    
        self.right_layout.addWidget(self.train_button)
        self.main_layout.addLayout(self.right_layout)

        self.main_window()

    def main_window(self):
        self.setWindowTitle(self.title)
        self.setLayout(self.main_layout)
        self.show()

    def launch_napari_window(self):                                       
        self.nap_win = NapariWindow(img_filename=self.cur_eval_img, 
                                    eval_data_path=self.eval_data_path, 
                                    train_data_path=self.train_data_path )
        self.nap_win.show()

    def item_eval_selected(self, item):
        self.cur_eval_img = item.data()

    def train_model(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec())
