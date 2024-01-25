import pytest
import sys
sys.path.append('../')

from PyQt5.QtCore import Qt

from dcp_client.gui.welcome_window import WelcomeWindow
from dcp_client.app import Application
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync
from dcp_client.utils import settings

@pytest.fixture
def setup_global_variable():
    settings.accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
    yield settings.accepted_types

@pytest.fixture
def app(qtbot):
    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path='.')
    application = Application(BentomlModel(), rsyncer, FilesystemImageStorage(), "0.0.0.0", 7010)
    # Create an instance of WelcomeWindow
    #q_app = QApplication([])
    widget = WelcomeWindow(application)
    qtbot.addWidget(widget)
    yield widget
    widget.close()

def test_welcome_window_initialization(app):
    assert app.title == "Select Dataset"
    assert app.val_textbox.text() == ""
    assert app.inprogr_textbox.text() == ""
    assert app.train_textbox.text() == ""
    
''''
# TODO wait for github respose
def test_browse_eval_clicked(qtbot, app, monkeypatch):
    # Mock the QFileDialog so that it immediately returns a directory
    def handle_dialog(*args, **kwargs):
        #if app.fd.isVisible(): 
        QCoreApplication.processEvents()
        app.app.eval_data_path = '/path/to/selected/directory'

    #def mock_file_dialog(*args, **kwargs):
    #       return ['/path/to/selected/directory']
    
    #monkeypatch.setattr(QFileDialog, 'getExistingDirectory', mock_file_dialog)
    QTimer.singleShot(100, handle_dialog)

    #monkeypatch.setattr(app, 'browse_eval_clicked', mock_file_dialog)
    #monkeypatch.setattr(QFileDialog, 'getExistingDirectory', mock_file_dialog)
    # Simulate clicking the browse button for evaluation directory
    qtbot.mouseClick(app.file_open_button_val, Qt.LeftButton, delay=1)
    # Check if the textbox is updated with the selected path
    assert app.val_textbox.text() == '/path/to/selected/directory'

def test_browse_eval_clicked(qtbot, app):
    # Simulate clicking the browse button for evaluation directory
    qtbot.mouseClick(app.file_open_button_val, Qt.LeftButton)
    # Check if the QFileDialog is shown
    assert qtbot.waitUntil(lambda: hasattr(app, 'app.eval_data_path'), timeout=1000)
    # Check if the textbox is updated with the selected path
    assert app.val_textbox.text() == app.app.eval_data_path


def test_browse_train_clicked(qtbot, app):
    # Simulate clicking the browse button for train directory
    qtbot.mouseClick(app.file_open_button_train, Qt.LeftButton)
    # Check if the QFileDialog is shown
    assert qtbot.waitUntil(lambda: hasattr(app, 'app.train_data_path'), timeout=1000)
    # Check if the textbox is updated with the selected path
    assert app.train_textbox.text() == app.app.train_data_path

def test_browse_inprogr_clicked(qtbot, app):
    # Simulate clicking the browse button for in-progress directory
    qtbot.mouseClick(app.file_open_button_prog, Qt.LeftButton)
    # Check if the QFileDialog is shown
    assert qtbot.waitUntil(lambda: hasattr(app, 'app.inprogr_data_path'), timeout=1000)
    # Check if the textbox is updated with the selected path
    assert app.inprogr_textbox.text() == app.app.inprogr_data_path

'''
def test_start_main_not_selected(qtbot, app):
    app.app.train_data_path = None
    app.app.eval_data_path = None
    app.sim = True
    qtbot.mouseClick(app.start_button, Qt.LeftButton)
    assert not hasattr(app, 'mw')

def test_start_main(qtbot, app, setup_global_variable):
    settings.accepted_types = setup_global_variable
    # Set some paths for testing
    app.app.eval_data_path = "/path/to/eval"
    app.app.train_data_path = "/path/to/train"
    # Simulate clicking the start button
    qtbot.mouseClick(app.start_button, Qt.LeftButton)
    # Check if the main window is created
    #assert qtbot.waitUntil(lambda: hasattr(app, 'mw'), timeout=1000)  
    assert hasattr(app, 'mw')
    # Check if the WelcomeWindow is hidden
    assert app.isHidden()

'''
def test_start_upload_and_main(qtbot, app, setup_global_variable):
    # TODO
'''