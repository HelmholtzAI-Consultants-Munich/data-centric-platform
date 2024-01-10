import os
import pytest
from skimage.io import imsave
from skimage import data

from dcp_client.utils.fsimagestorage import FilesystemImageStorage

@pytest.fixture
def fis():
    return FilesystemImageStorage()

@pytest.fixture
def sample_image():
    # Create a sample image
    img = data.astronaut()
    fname = 'test_img.png'
    imsave(fname, img)
    os.chmod(fname, 0o0777)
    return fname
    
def test_load_image(fis, sample_image):
    img_test = fis.load_image('.', sample_image)
    assert img_test.all() == data.astronaut().all()
    os.remove(sample_image)

def test_move_image(fis, sample_image):
    temp_dir = 'temp'
    os.mkdir(temp_dir)
    fis.move_image('.', temp_dir, sample_image)
    assert os.path.exists(os.path.join(temp_dir, 'test_img.png'))
    os.remove(os.path.join(temp_dir, 'test_img.png'))
    os.rmdir(temp_dir)

def test_save_image(fis):
    img = data.astronaut()
    fname = 'output.png'
    fis.save_image('.', fname, img)
    assert os.path.exists(fname)
    os.remove(fname)

def test_delete_image(fis, sample_image):
    temp_dir = 'temp'
    os.mkdir(temp_dir)
    fis.move_image('.', temp_dir, sample_image)
    fis.delete_image(temp_dir, 'test_img.png')
    assert not os.path.exists(os.path.join(temp_dir, 'test_img.png'))
    os.rmdir(temp_dir)
