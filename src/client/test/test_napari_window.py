import os

from skimage import data
from skimage.io import imsave
import numpy as np
import napari

import pytest
from PyQt5.QtWidgets import QApplication
from dcp_client.app import Application
from dcp_client.gui.napari_window import NapariWindow
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils import settings


def _patch_napari_qt_viewer_close_event(qt_viewer):
    """Patch closeEvent to ignore 'canvas already deleted' RuntimeError (napari teardown order)."""
    original_close_event = qt_viewer.closeEvent

    def safe_close_event(event):
        try:
            original_close_event(event)
        except RuntimeError as e:
            if "has been deleted" in str(e):
                event.accept()
            else:
                raise

    qt_viewer.closeEvent = safe_close_event


@pytest.fixture
def napari_window(qtbot):

    # img1 = data.astronaut()
    # img2 = data.coffee()
    img = data.cat()
    img_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    img_mask[50:50, 50:50] = 1
    img_mask[100:200, 100:200] = 2
    img_mask[200:300, 200:300] = 3
    # img3_mask = img2_mask.copy()

    if not os.path.exists("cur_data_path"):
        os.mkdir("cur_data_path")

    if not os.path.exists("in_prog"):
        os.mkdir("in_prog")

    if not os.path.exists("uncur_data_path"):
        os.mkdir("uncur_data_path")
    
    imsave("uncur_data_path/cat.png", img)
    imsave("uncur_data_path/cat_seg.tiff", img_mask)

    application = Application(
        BentomlModel(),
        1,
        FilesystemImageStorage(),
        "0.0.0.0",
        7010,
        os.path.join(os.getcwd(), "uncur_data_path"),
        os.path.join(os.getcwd(), "cur_data_path"),
        os.path.join(os.getcwd(), "in_prog"),
    )

    application.cur_selected_img = "cat.png"
    application.cur_selected_path = application.uncur_data_path

    widget = NapariWindow(application)
    qtbot.addWidget(widget)
    # Patch napari's qt_viewer closeEvent so teardown doesn't raise RuntimeError
    # when the canvas C++ object was already deleted (known napari/Qt order issue)
    if getattr(widget, "viewer", None) is not None and getattr(
        widget.viewer.window, "qt_viewer", None
    ) is not None:
        _patch_napari_qt_viewer_close_event(widget.viewer.window.qt_viewer)
    yield widget
    # Close napari viewer first, then clear reference so NapariWindow.closeEvent
    # does not try to close it again
    if getattr(widget, "viewer", None) is not None:
        try:
            widget.viewer.close()
        except RuntimeError:
            pass
        except Exception:
            pass
        widget.viewer = None
        QApplication.processEvents()
    widget.close()


def test_napari_window_initialization(napari_window):
    assert napari_window.viewer is not None
    if napari_window.app.num_classes > 1:
        assert napari_window.qctrl is not None
        assert napari_window.mask_choice_dropdown is not None


def test_on_add_to_curated_button_clicked(napari_window, monkeypatch):
    # Mock the create_warning_box method
    def mock_create_warning_box(message_text, message_title):
        return None

    monkeypatch.setattr(napari_window, "create_warning_box", mock_create_warning_box)

    napari_window.app.cur_selected_img = "cat.png"
    napari_window.app.cur_selected_path = napari_window.app.uncur_data_path

    napari_window.viewer.layers.selection.active.name = "cat_seg"

    # Simulate the button click
    napari_window.on_add_to_curated_button_clicked()

    assert not os.path.exists("uncur_data_path/cat.tiff")
    assert not os.path.exists("uncur_data_path/cat_seg.tiff")
    assert os.path.exists("cur_data_path/cat.png")
    assert os.path.exists("cur_data_path/cat_seg.tiff")
