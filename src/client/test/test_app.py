import os
from skimage import data
from skimage.io import imsave
import unittest

from dcp_client.app import Application
from dcp_client.utils.bentoml_model import BentomlModel
from dcp_client.utils.fsimagestorage import FilesystemImageStorage
from dcp_client.utils.sync_src_dst import DataRSync

class TestApplication(unittest.TestCase):
    
    def test_run_train(self):
        pass
    
    def test_run_inference(self):
        pass
    
    def test_load_image(self):

        img = data.astronaut()
        img2 = data.cat()
        os.mkdir('in_prog')
        imsave('in_prog/test_img.png', img)
        imsave('in_prog/test_img2.png', img2)
        rsyncer = DataRSync(user_name="local",
                          host_name="local",
                          server_repo_path='.')
        self.app = Application(BentomlModel(),
                               rsyncer,
                               FilesystemImageStorage(),
                               "0.0.0.0",
                               7010)
                        
        self.app.cur_selected_img = 'test_img.png'
        self.app.cur_selected_path = 'in_prog'

        img_test = self.app.load_image() # if image_name is None
        self.assertEqual(img.all(), img_test.all())
        img_test2 = self.app.load_image('test_img2.png') # if a filename is given
        self.assertEqual(img2.all(), img_test2.all())

        # delete everyting we created
        os.remove('in_prog/test_img.png')
        os.remove('in_prog/test_img2.png')
        os.rmdir('in_prog')

    def test_save_image(self):
         pass
    
    def test_move_images(self):
         pass
    
    def test_delete_images(self):
         pass
    
    def test_search_segs(self):
         pass
         
         
if __name__=='__main__':
	unittest.main()