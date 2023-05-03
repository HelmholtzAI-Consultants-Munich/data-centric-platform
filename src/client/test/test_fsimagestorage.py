import os
from skimage.io import imsave
from skimage import data
import unittest

from dcp_client.fsimagestorage import FilesystemImageStorage


class TestFilesystemImageStorage(unittest.TestCase):

    def test_load_image(self):
        fis = FilesystemImageStorage()
        img = data.astronaut()
        fname = 'test_img.png'
        imsave(fname, img)
        img_test = fis.load_image('.', fname)
        self.assertEqual(img.all(), img_test.all())
        os.remove(fname)

    def test_move_image(self):
        fis = FilesystemImageStorage()
        img = data.astronaut()
        fname = 'test_img.png'
        os.mkdir('temp')
        imsave(fname, img)
        fis.move_image('.', 'temp', fname)
        self.assertTrue(os.path.exists('temp/test_img.png'))
        os.remove('temp/test_img.png')
        os.rmdir('temp')

    def test_save_image(self):
        fis = FilesystemImageStorage()
        img = data.astronaut()
        fname = 'test_img.png'
        fis.save_image('.', fname, img)
        self.assertTrue(os.path.exists(fname))
        os.remove(fname)

    def test_delete_image(self):
        fis = FilesystemImageStorage()
        img = data.astronaut()
        fname = 'test_img.png'
        os.mkdir('temp')
        imsave('temp/test_img.png', img)
        fis.delete_image('temp', fname)
        self.assertFalse(os.path.exists('temp/test_img.png'))
        os.rmdir('temp')


if __name__=='__main__':
	unittest.main()