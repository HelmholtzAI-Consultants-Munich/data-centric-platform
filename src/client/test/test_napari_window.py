import os

from skimage import data
from skimage.io import imsave
import numpy as np

import pytest
from dcp_client.app import Application
from dcp_client.gui.napari_window import NapariWindow

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

    if not os.path.exists('train_data_path'): 
        os.mkdir('train_data_path')
        imsave('train_data_path/astronaut.png', img1)

    if not os.path.exists('in_prog'): 
        os.mkdir('in_prog')
        imsave('in_prog/coffee.png', img2)

    if not os.path.exists('eval_data_path'): 
        os.mkdir('eval_data_path')
        imsave('eval_data_path/cat.png', img3)
        imsave('eval_data_path/cat_seg.png', img3)
    
    imsave('eval_data_path/cat_test.png', img3)
    imsave('eval_data_path/cat_test_seg.png', img3)

    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path='.')
    application = Application(
        BentomlModel(), 
        rsyncer, 
        FilesystemImageStorage(), 
        "0.0.0.0", 
        7010,
        os.path.join(os.getcwd(), 'eval_data_path'),
        os.path.join(os.getcwd(), 'train_data_path'),
        os.path.join(os.getcwd(), 'in_prog')
    )

    application.cur_selected_img = 'cat_test.png'
    application.cur_selected_path = application.eval_data_path

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
    assert napari_window.active_mask 

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

    napari_window.switch_user_mask()
    assert napari_window.active_mask is True


def test_get_position_label(napari_window):
    napari_window.event_coords = (0, 10, 20)
    result = napari_window.get_position_label(napari_window.layer.data)
    assert result is not None
  

def test_update_source_mask_new_color(napari_window):
    source_mask = np.zeros((1, 3, 3))  
    mask_fill = np.ones((3, 3), dtype=bool)  
    c = 1
    label = 5
    label_seg = 10

    result = napari_window.update_source_mask(source_mask, mask_fill, c, label, label_seg)
    assert np.array_equal(result, 5*np.ones_like(result))


def test_on_add_to_curated_button_clicked(napari_window, monkeypatch):
    # Mock the create_warning_box method
    def mock_create_warning_box(message_text, message_title):
        return None  

    monkeypatch.setattr(napari_window, 'create_warning_box', mock_create_warning_box)

    # assert napari_window.app.cur_selected_path == 'eval_data_path'

    napari_window.app.cur_selected_img = 'cat_test.png'
    napari_window.app.cur_selected_path = napari_window.app.eval_data_path

    napari_window.viewer.layers.selection.active.name = 'cat_test_seg' 

    # Simulate the button click
    napari_window.on_add_to_curated_button_clicked()

    assert os.path.exists('train_data_path/cat_test_seg.tiff')
    assert os.path.exists('train_data_path/cat_test.png')
    assert not os.path.exists('eval_data_path/cat_test.png')

# @pytest.fixture(scope='session', autouse=True)
# def cleanup_files(request):
#     # This code runs after all tests from all files have completed
#     yield
#     # Clean up
#     for fname in os.listdir('train_data_path'):
#         os.remove(os.path.join('train_data_path', fname))
#     os.rmdir('train_data_path')

#     for fname in os.listdir('in_prog'):
#         os.remove(os.path.join('in_prog', fname))
#     os.rmdir('in_prog')

#     for fname in os.listdir('eval_data_path'):
#         os.remove(os.path.join('eval_data_path', fname))
#     os.rmdir('eval_data_path')