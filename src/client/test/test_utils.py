import sys

sys.path.append("../")
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
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        filepath = '\\here\\we\\are\\testing\\something.txt'
        assert utils.get_path_parent(filepath)== '\\here\\we\\are\\testing'
    else:
        filepath = '/here/we/are/testing/something.txt'
        assert utils.get_path_parent(filepath)== '/here/we/are/testing'

def test_join_path():
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        filepath = '\\here\\we\\are\\testing\\something.txt'
        path1 = '\\here\\we\\are\\testing'
        path2 = 'something.txt'
    else:
        filepath = '/here/we/are/testing/something.txt'
        path1 = '/here/we/are/testing'
        path2 = 'something.txt'
    assert utils.join_path(path1, path2) == filepath


