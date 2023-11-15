import os
import sys
from skimage import data
from skimage.io import imsave
import pytest

sys.path.append("../")

from dcp_client.app import Application
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync


@pytest.fixture
def app():
    img = data.astronaut()
    img2 = data.cat()
    os.mkdir('in_prog')

    imsave('in_prog/test_img.png', img)
    imsave('in_prog/test_img2.png', img2)

    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path='.')
    app = Application(BentomlModel(), rsyncer, FilesystemImageStorage(), "0.0.0.0", 7010)

    app.cur_selected_img = 'test_img.png'
    app.cur_selected_path = 'in_prog'

    return app, img, img2

def test_load_image(app):
    app, img, img2 = app  # Unpack the app, img, and img2 from the fixture

    img_test = app.load_image()  # if image_name is None
    assert img.all() == img_test.all()

    img_test2 = app.load_image('test_img2.png')  # if a filename is given
    assert img2.all() == img_test2.all()

    # delete everything we created
    os.remove('in_prog/test_img.png')
    os.remove('in_prog/test_img2.png')
    os.rmdir('in_prog')

def test_run_train():
    pass

def test_run_inference():
    pass

def test_save_image():
    pass

def test_move_images():
    pass

def test_delete_images():
    pass

def test_search_segs():
    pass





         
         
