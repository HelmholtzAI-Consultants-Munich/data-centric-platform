from __future__ import annotations

from typing import TYPE_CHECKING

import napari
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from dcp_client.app import Application



class NapariWindow(QWidget):
    '''Napari Window Widget object.
    Opens the napari image viewer to view and fix the labeles.
    :param img_filename:
    :type img_filename: string
    :param eval_data_path:
    :type eval_data_path:
    :param train_data_path:
    :type train_data_path:
    '''

    def __init__(self, app: Application):
        super().__init__()
        
        self.app = app        
        self.setWindowTitle("napari viewer")
        self.viewer = napari.Viewer(show=False)


        img, labels = self.app.load_image_and_seg()
        self.viewer.add_image(img)
        if labels is not None: 
            self.viewer.add_labels(labels)

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


    @property
    def seg_layer_name(self) -> str:
        label_layers = [layer.name for layer in self.viewer.layers if type(layer) == napari.layers.Labels]
        return label_layers[0]

    @property
    def seg(self) -> NDArray:
        return self.viewer.layers[self.seg_layer_name].data


    def on_add_to_curated_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        self.app.save_seg(self.seg)
        self.close()

    def on_add_to_inprogress_button_clicked(self):
        '''
        Defines what happens when the button is clicked.
        '''
        self.app.set_seg_name(self.seg_layer_name)
        self.app.save2(self.seg)
        self.close()
