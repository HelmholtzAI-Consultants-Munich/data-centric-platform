from skimage.io import imread, imsave
import os
from pathlib import Path

from app import ImageStorage


class FilesystemImageStorage(ImageStorage):

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
