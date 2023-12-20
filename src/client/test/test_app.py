import os
import sys
sys.path.append("../")
import pytest
import subprocess
import time

from skimage import data
from skimage.io import imsave

from dcp_client.app import Application
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync

@pytest.fixture
def app():
    img = data.astronaut()
    img2 = data.cat()
    if not os.path.exists('in_prog'): 
        os.mkdir('in_prog')
        imsave('in_prog/test_img.png', img)
        imsave('in_prog/test_img2.png', img2)

    if not os.path.exists('eval_data_path'): 
        os.mkdir('eval_data_path')
        imsave('eval_data_path/test_img.png', img)

    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path='.')
    app = Application(BentomlModel(),
                      rsyncer,
                      FilesystemImageStorage(),
                      "0.0.0.0",
                      7010,
                      os.path.join(os.getcwd(), 'eval_data_path'))
    
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

def test_run_inference_no_connection(app):
    app, _, _ = app 
    message_text, message_title = app.run_inference()
    assert message_text=="Connection could not be established. Please check if the server is running and try again."
    assert message_title=="Warning"

def test_run_inference_run(app):
    app, _, _ = app 
    # start the sevrer in the background locally
    command = [
        "bentoml",
        "serve", 
        '--working-dir', 
        '../server/dcp_server',
        "service:svc",
        "--reload",
        "--port=7010",
    ]
    process = subprocess.Popen(command)
    time.sleep(60) # and wait until it is setup
    # then do model serving
    message_text, message_title = app.run_inference()
    assert message_text== "Success! Masks generated for all images"
    assert message_title=="Information"
    os.remove('eval_data_path/test_img.png')
    os.remove('eval_data_path/test_img_seg.tiff')
    os.rmdir('eval_data_path')
    process.terminate()
    process.wait()
    process.kill()

'''
def test_run_train():
    

def test_save_image():
    pass

def test_move_images():
    pass

def test_delete_images():
    pass

def test_search_segs():
    pass

'''

         
         
