import os
from pathlib import Path

from skimage.io import imread, imsave
from bentoml_model import BentomlModel
import settings


class Application:

    def __init__(
        self, 
        bentoml_model: BentomlModel,
        eval_data_path: str = '', 
        train_data_path = '', 
        inprogr_data_path = '',     
        
    ):
        self.bentoml_model = bentoml_model
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path
        self.cur_selected_img = ''
        
    def run_train(self):
        return self.bentoml_model.run_train(self.train_data_path)
    
    def run_inference(self):
        list_of_files_not_suported = self.bentoml_model.run_inference(self.eval_data_path)
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
        self.potential_seg_name = Path(self.cur_selected_img).stem + '_seg.tiff' #+Path(self.img_filename).suffix
        if os.path.exists(os.path.join(self.eval_data_path, self.cur_selected_img)):
            img = imread(os.path.join(self.eval_data_path, self.cur_selected_img))
            if os.path.exists(os.path.join(self.eval_data_path, self.potential_seg_name)):
                seg = imread(os.path.join(self.eval_data_path, self.potential_seg_name))
            else: seg = None
        else: 
            img = imread(os.path.join(self.train_data_path, self.cur_selected_img))
            if os.path.exists(os.path.join(self.train_data_path, self.potential_seg_name)):
                seg = imread(os.path.join(self.train_data_path, self.potential_seg_name))
            else: seg = None
        return img, seg
    
    def save_seg(self, seg, from_directory, to_directory):
        
        os.replace(os.path.join(from_directory, self.cur_selected_img), os.path.join(to_directory, self.cur_selected_img))
        seg_name = Path(self.cur_selected_img).stem+ '_seg.tiff' #+Path(self.img_filename).suffix
        imsave(os.path.join(to_directory, seg_name), seg)
        if os.path.exists(os.path.join(from_directory, seg_name)): 
            os.remove(os.path.join(from_directory, seg_name))


