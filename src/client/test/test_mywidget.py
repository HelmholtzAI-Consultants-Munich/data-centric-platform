import pytest
import sys
sys.path.append('../')

from PyQt5.QtWidgets import QMessageBox

from dcp_client.gui._my_widget import MyWidget

@pytest.fixture
def app(qtbot):
    #q_app = QApplication([])
    widget = MyWidget()
    qtbot.addWidget(widget)
    yield widget
    widget.close()

def test_create_warning_box_ok(qtbot, app):
    result = None
    app.sim = True
    def execute_warning_box():
        nonlocal result
        box = QMessageBox()
        result = app.create_warning_box("Test Message", custom_dialog=box)
    qtbot.waitUntil(execute_warning_box, timeout=5000) 
    assert result is True  

def test_create_warning_box_cancel(qtbot, app):
    result = None
    app.sim = True
    def execute_warning_box():
        nonlocal result
        box = QMessageBox()
        result = app.create_warning_box("Test Message", add_cancel_btn=True, custom_dialog=box)
    qtbot.waitUntil(execute_warning_box, timeout=5000)  # Add a timeout for the function to execute
    assert result is False  
