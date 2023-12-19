import sys
sys.path.append("../")
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QMessageBox
from qtpy.QtCore import Qt, QTimer
from dcp_client.utils import utils

def test_get_relative_path():
    filepath = '/here/we/are/testing/something.txt'
    assert utils.get_relative_path(filepath)== 'something.txt'

def test_get_path_stem():
    filepath = '/here/we/are/testing/something.txt'
    assert utils.get_path_stem(filepath)== 'something'

def test_get_path_name():
    filepath = '/here/we/are/testing/something.txt'
    assert utils.get_path_name(filepath)== 'something.txt'

def test_get_path_parent():
    filepath = '/here/we/are/testing/something.txt'
    assert utils.get_path_parent(filepath)== '/here/we/are/testing'

def test_join_path():
    filepath = '/here/we/are/testing/something.txt'
    path1 = '/here/we/are/testing'
    path2 = 'something.txt'
    assert utils.join_path(path1, path2) == filepath

def test_create_warning_box_ok(qtbot):
    result = None
    def execute_warning_box():
        nonlocal result
        box = QMessageBox()
        result = utils.create_warning_box("Test Message", custom_dialog=box, sim=True)
    qtbot.waitUntil(execute_warning_box, timeout=5000) 
    assert result is True  

def test_create_warning_box_cancel(qtbot):
    result = None
    def execute_warning_box():
        nonlocal result
        box = QMessageBox()
        result = utils.create_warning_box("Test Message", add_cancel_btn=True, custom_dialog=box, sim=True)
    qtbot.waitUntil(execute_warning_box, timeout=5000)  # Add a timeout for the function to execute
    assert result is False  