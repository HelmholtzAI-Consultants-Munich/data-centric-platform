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

    # img1 = data.astronaut()
    # img2 = data.coffee()
    img = data.cat()
    img_mask = np.zeros((2, img.shape[0], img.shape[1]), dtype=np.uint8)
    img_mask[0, 50:50, 50:50] = 1
    img_mask[1, 50:50, 50:50] = 1
    img_mask[0, 100:200, 100:200] = 2
    img_mask[1, 100:200, 100:200] = 1
    img_mask[0, 200:300, 200:300] = 3
    img_mask[1, 200:300, 200:300] = 2
    # img3_mask = img2_mask.copy()

    if not os.path.exists("train_data_path"):
        os.mkdir("train_data_path")

    if not os.path.exists("in_prog"):
        os.mkdir("in_prog")

    if not os.path.exists("eval_data_path"):
        os.mkdir("eval_data_path")
        imsave("eval_data_path/cat.png", img)

    imsave("eval_data_path/cat_seg.tiff", img_mask)

    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path=".")
    application = Application(
        BentomlModel(),
        rsyncer,
        FilesystemImageStorage(),
        "0.0.0.0",
        7010,
        os.path.join(os.getcwd(), "eval_data_path"),
        os.path.join(os.getcwd(), "train_data_path"),
        os.path.join(os.getcwd(), "in_prog"),
    )

    application.cur_selected_img = "cat.png"
    application.cur_selected_path = application.eval_data_path

    widget = NapariWindow(application)
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def test_napari_window_initialization(napari_window):
    assert napari_window.viewer is not None
    assert napari_window.qctrl is not None
    assert napari_window.mask_choice_dropdown is not None


def test_on_add_to_curated_button_clicked(napari_window, monkeypatch):
    # Mock the create_warning_box method
    def mock_create_warning_box(message_text, message_title):
        return None

    monkeypatch.setattr(napari_window, "create_warning_box", mock_create_warning_box)

    napari_window.app.cur_selected_img = "cat.png"
    napari_window.app.cur_selected_path = napari_window.app.eval_data_path

    napari_window.viewer.layers.selection.active.name = "cat_seg"

    # Simulate the button click
    napari_window.on_add_to_curated_button_clicked()

    assert not os.path.exists("eval_data_path/cat.tiff")
    assert not os.path.exists("eval_data_path/cat_seg.tiff")
    assert os.path.exists("train_data_path/cat.png")
    assert os.path.exists("train_data_path/cat_seg.tiff")
