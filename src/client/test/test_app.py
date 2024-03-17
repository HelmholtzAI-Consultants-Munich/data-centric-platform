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
    img1 = data.astronaut()
    img2 = data.coffee()
    img3 = data.cat()

    if not os.path.exists("in_prog"):
        os.mkdir("in_prog")
        imsave("in_prog/coffee.png", img2)

    if not os.path.exists("eval_data_path"):
        os.mkdir("eval_data_path")
        imsave("eval_data_path/cat.png", img3)

    rsyncer = DataRSync(user_name="local", host_name="local", server_repo_path=".")
    app = Application(
        BentomlModel(),
        rsyncer,
        FilesystemImageStorage(),
        "0.0.0.0",
        7010,
        os.path.join(os.getcwd(), "eval_data_path"),
    )

    return app, img1, img2, img3


def test_load_image(app):
    app, img, img2, _ = app  # Unpack the app, img, and img2 from the fixture

    app.cur_selected_img = "coffee.png"
    app.cur_selected_path = "in_prog"

    img_test = app.load_image()  # if image_name is None
    assert img.all() == img_test.all()

    app.cur_selected_path = "eval_data_path"
    img_test2 = app.load_image("cat.png")  # if a filename is given
    assert img2.all() == img_test2.all()


def test_run_inference_no_connection(app):
    app, _, _, _ = app
    message_text, message_title = app.run_inference()
    assert (
        message_text
        == "Connection could not be established. Please check if the server is running and try again."
    )
    assert message_title == "Warning"


def test_run_inference_run(app):
    app, _, _, _ = app
    # start the sevrer in the background locally
    command = [
        "bentoml",
        "serve",
        "--working-dir",
        "../server/dcp_server",
        "service:svc",
        "--reload",
        "--port=7010",
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, shell=False)
    # and wait until it is setup
    if sys.platform == "win32" or sys.platform == "cygwin":
        time.sleep(240)
    else:
        time.sleep(60)
    # then do model serving
    message_text, message_title = app.run_inference()
    # and assert returning message
    print(f"HERE: {message_text, message_title}")
    assert message_text == "Success! Masks generated for all images"
    assert message_title == "Information"
    # finally clean up process
    process.terminate()
    process.wait()
    process.kill()


def test_search_segs(app):
    app, _, _, _ = app
    app.cur_selected_img = "cat.png"
    app.cur_selected_path = "eval_data_path"
    app.search_segs()
    res = app.seg_filepaths
    assert len(res) == 1
    assert res[0] == "cat_seg.tiff"
    # also remove the seg as it is not needed for other scripts
    os.remove("eval_data_path/cat_seg.tiff")


"""
def test_run_train():
    pass

def test_save_image():
    pass

def test_move_images():
    pass

def test_delete_images():
    pass

"""
