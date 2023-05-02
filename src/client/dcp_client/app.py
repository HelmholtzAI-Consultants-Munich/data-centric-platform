from dcp_client import settings
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray
import os

from dcp_client import utils


class Model(ABC):
    @abstractmethod
    def run_train(self, path: str) -> None:
        pass
    
    @abstractmethod
    def run_inference(self, path: str) -> None:
        pass


class ImageStorage(ABC):
    @abstractmethod
    def load_image(self, from_directory, cur_selected_img) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def save_image(self, to_directory, cur_selected_img, img) -> None:
        pass

    def search_segs(self, cur_selected_img):
        """Returns a list of full paths of segmentations for an image"""
        # Check the directory the image was selected from:
        img_directory = utils.get_path_parent(cur_selected_img)
        # Take all segmentations of the image from the current directory:
        search_string = utils.get_path_stem(cur_selected_img) + '_seg'
        seg_files = [file_name for file_name in os.listdir(img_directory) if search_string in file_name]
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
        self.cur_selected_path = ''
        self.seg_filepaths = []
    
    def run_train(self):
        """ Checks if the ml model is connected to the server, connects if not (and if possible), and trains the model with all data available in train_data_path """
        if not self.ml_model.is_connected:
            connection_success = self.ml_model.connect(ip=self.server_ip, port=self.server_port)
            if not connection_success: return "Connection could not be established. Please check if the server is running and try again."
        return self.ml_model.run_train(self.train_data_path)
    
    def run_inference(self):
        """ Checks if the ml model is connected to the server, connects if not (and if possible), and runs inference on all images in eval_data_path """
        if not self.ml_model.is_connected:
            connection_success = self.ml_model.connect(ip=self.server_ip, port=self.server_port)
            if not connection_success: 
                message_text = "Connection could not be established. Please check if the server is running and try again."
                return message_text, "Warning"
        list_of_files_not_suported = self.ml_model.run_inference(self.eval_data_path)
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            message_text = "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: " + ", ".join(settings.accepted_types)+ ". The files that were not supported are: " + ", ".join(list_of_files_not_suported)
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
    
    def save_image(self, dst_directory, image_name, img):
        """ Saves img array image in the dst_directory with filename cur_selected_img """
        self.fs_image_storage.save_image(dst_directory, image_name, img)

    def move_images(self, dst_directory, move_segs=False):
        """ Moves cur_selected_img image from the current directory to the dst_directory """
        #if image_name is None:
        self.fs_image_storage.move_image(self.cur_selected_path, dst_directory, self.cur_selected_img)
        if move_segs:
            for seg_name in self.seg_filepaths:
                self.fs_image_storage.move_image(self.cur_selected_path, dst_directory, seg_name)

    def delete_images(self, image_names):
        """ If image_name in the image_names list exists in the current directory it is deleted """
        for image_name in image_names:
            if os.path.exists(os.path.join(self.cur_selected_path, image_name)):    
                self.fs_image_storage.delete_image(self.cur_selected_path, image_name)

    def search_segs(self):
        """  Searches in cur_selected_path for all possible segmentation files associated to cur_selected_img.
            These files should have a _seg extension to the cur_selected_img filename. """
        self.seg_filepaths = self.fs_image_storage.search_seg(self.cur_selected_path, self.cur_selected_img)


