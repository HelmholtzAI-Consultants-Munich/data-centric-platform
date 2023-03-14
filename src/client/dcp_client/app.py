import os
from pathlib import Path

from skimage.io import imread, imsave

class Napari_Application():
    '''Contains functions with main code'''

    def __init__(self, 
                img_filename,
                eval_data_path,
                train_data_path,
                inprogr_data_path):
        self.img_filename = img_filename
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path

    def load_image_seg(self):
        self.potential_seg_name = Path(self.img_filename).stem + '_seg.tiff' #+Path(self.img_filename).suffix
        if os.path.exists(os.path.join(self.eval_data_path, self.img_filename)):
            img = imread(os.path.join(self.eval_data_path, self.img_filename))
            if os.path.exists(os.path.join(self.eval_data_path, self.potential_seg_name)):
                seg = imread(os.path.join(self.eval_data_path, self.potential_seg_name))
            else: seg = None
        else: 
            img = imread(os.path.join(self.train_data_path, self.img_filename))
            if os.path.exists(os.path.join(self.train_data_path, self.potential_seg_name)):
                seg = imread(os.path.join(self.train_data_path, self.potential_seg_name))
            else: seg = None
        return img, seg
    
    def save_seg(self, seg, from_directory, to_directory):
        
        os.replace(os.path.join(from_directory, self.img_filename), os.path.join(to_directory, self.img_filename))
        seg_name = Path(self.img_filename).stem+ '_seg.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(to_directory, seg_name), seg)
        if os.path.exists(os.path.join(from_directory, seg_name)): 
            os.remove(os.path.join(from_directory, seg_name))
