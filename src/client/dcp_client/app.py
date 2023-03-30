import os
from pathlib import Path

from skimage.io import imread, imsave
import asyncio
from bentoml.client import Client

import settings


class Application:

    def __init__(
        self, 
        eval_data_path = '', 
        train_data_path = '', 
        inprogr_data_path = '',
    ):
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path
        self.client = Client.from_url("http://0.0.0.0:7010") # have the url of the bentoml service here
        self.cur_selected_img = ''
        self.fs_image_storage = FilesystemImageStorage()

    async def _run_train(self):
        response = await self.client.async_retrain(self.train_data_path)
        return response
    
    def run_train(self):
        return asyncio.run(self._run_train())
    
    async def _run_inference(self):
        response = await self.client.async_segment_image(self.eval_data_path)
        return response
    
    def run_inference(self):
        list_of_files_not_suported = asyncio.run(self._run_inference())
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            message_text = "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: ", settings.accepted_types, "The files that were not supported are: " + ", ".join(list_of_files_not_suported)
            message_title = "Warning"
        else:
            message_text = "Success! Masks generated for all images"
            message_title="Success"
        return message_text, message_title
    

    def load_image_seg(self):
        return self.fs_image_storage.load_image_seg(
            eval_data_path=self.eval_data_path,
            train_data_path=self.train_data_path,
            cur_selected_img = self.cur_selected_img,
        )

    
    def save_seg(self, seg, from_directory, to_directory):
        self.fs_image_storage.save_seg(seg, from_directory, to_directory, self.cur_selected_img)

       


class FilesystemImageStorage:

    def load_image_seg(self, eval_data_path, train_data_path, cur_selected_img):
        
        potential_seg_name = Path(cur_selected_img).stem + '_seg.tiff' #+Path(self.img_filename).suffix
        if os.path.exists(os.path.join(eval_data_path, cur_selected_img)):
            img = imread(os.path.join(eval_data_path, cur_selected_img))
            if os.path.exists(os.path.join(eval_data_path, potential_seg_name)):
                seg = imread(os.path.join(eval_data_path, potential_seg_name))
            else: seg = None
        else: 
            img = imread(os.path.join(train_data_path, cur_selected_img))
            if os.path.exists(os.path.join(train_data_path, potential_seg_name)):
                seg = imread(os.path.join(train_data_path, potential_seg_name))
            else: seg = None
        return img, seg      
    
    def save_seg(self, seg, from_directory, to_directory, cur_selected_img):
        os.replace(os.path.join(from_directory, cur_selected_img), os.path.join(to_directory, cur_selected_img))
        seg_name = Path(cur_selected_img).stem+ '_seg.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(to_directory, seg_name), seg)
        if os.path.exists(os.path.join(from_directory, seg_name)): 
            os.remove(os.path.join(from_directory, seg_name))   