import os

from skimage import data
from skimage.io import imsave

import pytest
from napari.qt import QtViewer
from dcp_client.app import Application
from dcp_client.gui.napari_window import NapariWindow
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QMessageBox

from dcp_client.gui.welcome_window import WelcomeWindow
from dcp_client.app import Application
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync
from dcp_client.utils import settings

# @pytest.fixture
# def napari_app():
#     app = Application([])
#     napari_app = QtViewer()
#     yield napari_app
#     napari_app.close()

@pytest.fixture
def napari_window(qtbot):

    img1 = data.astronaut()
    img2 = data.coffee()
    img3 = data.cat()

    if not os.path.exists('in_prog'): 
        os.mkdir('in_prog')
        imsave('in_prog/coffee.png', img2)

    if not os.path.exists('eval_data_path'): 
        os.mkdir('eval_data_path')
        imsave('eval_data_path/cat.png', img3)

    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path='.')
    application = Application(
        BentomlModel(), 
        rsyncer, 
        FilesystemImageStorage(), 
        "0.0.0.0", 
        7010,
        os.path.join(os.getcwd(), 'eval_data_path')
    )

    application.cur_selected_img = 'cat.png'
    application.cur_selected_path = 'eval_data_path'

    widget = NapariWindow(application)
    qtbot.addWidget(widget) 
    yield widget 
    widget.close()


def test_napari_window_initialization(napari_window):
    assert napari_window.viewer is not None
    assert napari_window.qctrl is not None
    assert napari_window.mask_choice_dropdown is not None

def test_switch_to_active_mask(napari_window):
    napari_window.switch_to_active_mask()
    assert napari_window.active_mask is True
 

def test_switch_to_non_active_mask(napari_window):
    napari_window.switch_to_non_active_mask()
    assert napari_window.active_mask is False
   

def test_set_active_mask(napari_window):
    napari_window.active_mask_index = 0
    napari_window.set_active_mask()
    assert napari_window.active_mask is True

    napari_window.active_mask_index = 1
    napari_window.set_active_mask()
    assert napari_window.active_mask is False
   

def test_save_click_coordinates(napari_window):
 
    event_position = (0, 10, 20)
    result = napari_window.save_click_coordinates(event_position)
    assert result is not None
  

def test_switch_user_mask(napari_window):
  
    # napari_window.viewer.dims.current_step[0] = 1
    napari_window.switch_user_mask()
    assert napari_window.active_mask is False


def test_get_position_label(napari_window):
    napari_window.event_coords = (0, 10, 20)
    result = napari_window.get_position_label(napari_window.layer.data)
    assert result is not None
  

