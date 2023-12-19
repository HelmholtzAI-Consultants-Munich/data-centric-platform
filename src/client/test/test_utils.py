import sys
sys.path.append("../")
#from qtpy.QtTest import QTest
#from qtpy.QtWidgets import QMessageBox
#from qtpy.QtCore import Qt
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

'''
def test_create_warning_box_ok(qtbot):
    # Using qtbot fixture provided by pytest-qt
    result = None

    def execute_warning_box():
        nonlocal result
        box = QMessageBox()
        while result is None:
            result = utils.create_warning_box("Test Message", custom_dialog=box)
        # Close the message box by simulating a click on the OK button
        QTest.mouseClick(box.button(QMessageBox.Ok), Qt.LeftButton)

    # Run the function in the main thread using qtbot
    #qtbot.addWidget(box)  # Add the QMessageBox widget to the QtBot's widget registry
    qtbot.waitUntil(execute_warning_box, timeout=5000) #lambda: result is not None)

    assert result is True  # Assert the expected result


def test_create_warning_box_cancel(qtbot):
    # Using qtbot fixture provided by pytest-qt
    result = None

    def execute_warning_box():
        nonlocal result
        result = utils.create_warning_box("Test Message", add_cancel_btn=True)
        # Close the message box by simulating a click on the Cancel button
        QTest.mouseClick(box, Qt.LeftButton)

    box = QMessageBox()
    box.button(QMessageBox.Cancel)
    # Run the function in the main thread using qtbot
    qtbot.addWidget(box)  # Add the QMessageBox widget to the QtBot's widget registry
    qtbot.waitUntil(execute_warning_box)  # Add a timeout for the function to execute
        
    assert result is False  # Assert the expected result
'''