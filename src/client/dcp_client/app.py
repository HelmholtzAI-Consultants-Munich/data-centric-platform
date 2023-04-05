import settings
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray
import os

import utils

class Model(ABC):
    @abstractmethod
    def run_train(self, path: str) -> None:
        pass
    
    @abstractmethod
    def run_inference(self, path: str) -> None:
        pass


class ImageStorage(ABC):
    @abstractmethod
    def load_image(self, cur_selected_img) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def save_image(self, seg, from_directory, to_directory, cur_selected_img) -> None:
        pass

    def search_segs(self, cur_selected_img):
        """Returns a list of full paths of segmentations for an image"""
        # Check the directory the image was selected from:
        img_directory = utils.get_path_parent(cur_selected_img)
        # Take all segmentations of the image from the current directory:
        search_string = utils.get_path_stem(cur_selected_img) + '_seg'
        seg_files = [os.path.join(img_directory, file_name) for file_name in os.listdir(img_directory) if search_string in file_name]
        return seg_files


class Application:
    def __init__(
        self, 
        ml_model: Model,
        image_storage: ImageStorage,
        server_ip: str = '0.0.0.0',
        server_port: int = 7010,
        eval_data_path: str = '', 
        train_data_path: str = '', 
        inprogr_data_path: str = '',     
    ):
        self.ml_model = ml_model
        self.fs_image_storage = image_storage
        self.server_ip = server_ip
        self.server_port = server_port
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path
        self.cur_selected_img = ''
       
    
    def run_train(self):
        if not self.ml_model.is_connected:
            connection_success = self.ml_model.connect(ip=self.server_ip, port=self.server_port)
            if not connection_success: return "Connection could not be established. Please check if the server is running and try again."
        return self.ml_model.run_train(self.train_data_path)
    
    def run_inference(self):
        if not self.ml_model.is_connected:
            connection_success = self.ml_model.connect(ip=self.server_ip, port=self.server_port)
            if not connection_success: 
                message_text = "Connection could not be established. Please check if the server is running and try again."
                return message_text, "Warning"
        list_of_files_not_suported = self.ml_model.run_inference(self.eval_data_path)
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            message_text = "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: ", settings.accepted_types, "The files that were not supported are: " + ", ".join(list_of_files_not_suported)
            message_title = "Warning"
        else:
            message_text = "Success! Masks generated for all images"
            message_title="Success"
        return message_text, message_title

    def load_image(self, image_path=None):
        if image_path is None:
            return self.fs_image_storage.load_image(self.cur_selected_img)
        else: return self.fs_image_storage.load_image(image_path)
    
    def search_segs(self):
        return self.fs_image_storage.search_segs(self.cur_selected_img)
    
    def save_image(self, to_directory, cur_selected_img, img):
        self.fs_image_storage.save_image(to_directory, cur_selected_img, img)

    def move_image(self, from_directory, to_directory, cur_selected_img):
        self.fs_image_storage.move_image(from_directory, to_directory, cur_selected_img)

    def delete_image(self, from_directory, cur_selected_img):
        if os.path.exists(os.path.join(from_directory, cur_selected_img)):    
            self.fs_image_storage.delete_image(from_directory, cur_selected_img)


