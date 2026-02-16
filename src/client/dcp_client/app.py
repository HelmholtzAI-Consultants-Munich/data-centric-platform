import os
from abc import ABC, abstractmethod
from typing import Tuple
from numpy.typing import NDArray

from dcp_client.utils import utils
from dcp_client.utils import settings

from dcp_client.utils.logger import get_logger

logger = get_logger(__name__)


class Model(ABC):
    @abstractmethod
    def run_train(self, path: str) -> None:
        pass

    @abstractmethod
    async def segment_image(self, image: NDArray) -> NDArray:
        """Segments a single image.
        
        :param image: Pre-loaded image as numpy array
        :type image: NDArray
        :return: Segmentation mask
        :rtype: NDArray
        """
        pass


class ImageStorage(ABC):
    @abstractmethod
    def load_image(self, from_directory, cur_selected_img) -> Tuple[NDArray, NDArray]:
        pass

    @abstractmethod
    def save_image(self, to_directory, cur_selected_img, img) -> None:
        pass

    def search_segs(self, img_directory, cur_selected_img):
        """Returns a list of full paths of segmentations for an image"""
        # Take all segmentations of the image from the current directory:
        search_string = utils.get_path_stem(cur_selected_img) + "_seg"
        seg_files = [
            file_name
            for file_name in os.listdir(img_directory)
            if (
                search_string == utils.get_path_stem(file_name)
                or str(file_name).startswith(search_string)
            )
        ]
        return seg_files


class Application:
    def __init__(
        self,
        ml_model: Model,
        num_classes: int,
        image_storage: ImageStorage,
        server_ip: str,
        server_port: int,
        eval_data_path: str = "",
        train_data_path: str = "",
        inprogr_data_path: str = "",
    ):
        self.ml_model = ml_model
        self.num_classes = num_classes
        self.fs_image_storage = image_storage
        self.server_ip = server_ip
        self.server_port = server_port
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.inprogr_data_path = inprogr_data_path
        self.cur_selected_img = ""
        self.cur_selected_path = ""
        self.seg_filepaths = []

    def try_server_connection(self):
        """
        Checks if the ml model is connected to server and attempts to connect if not.
        """
        connection_success = self.ml_model.connect(
            ip=self.server_ip, port=self.server_port
        )
        return connection_success

    def run_train(self):
        """Checks if the ml model is connected to the server, connects if not (and if possible), and trains the model with all data available in train_data_path"""
        if not self.ml_model.is_connected and not self.try_server_connection():
            message_title = "Warning"
            message_text = "Connection could not be established. Please check if the server is running and try again."
            return message_text, message_title
        message_title = "Information"
        message_text = self.ml_model.run_train(self.train_data_path)
        if message_text is None:
            message_text = "An error has occured on the server. Please check your image data and configurations. If the problem persists contact your software provider."
            message_title = "Error"
        return message_text, message_title

    async def _segment_images_recursively(self, image_list, output_directory):
        """Recursively segments each image in the list.
        
        :param image_list: List of image file paths in eval_data_path
        :type image_list: list
        :param output_directory: Directory where segmentation masks will be saved
        :type output_directory: str
        :return: List of unsupported files
        :rtype: list
        """
        unsupported_files = []
        for img_path in image_list:
            try:
                # Load the image
                img = self.fs_image_storage.load_image(self.eval_data_path, os.path.basename(img_path))
                
                # Segment the image
                mask = await self.ml_model.segment_image(img)
                
                # Save the segmentation mask
                seg_name = (
                    utils.get_path_stem(os.path.basename(img_path))
                    + settings.seg_name_string
                    + ".tiff"
                )
                self.fs_image_storage.save_image(output_directory, seg_name, mask)
            except Exception as e:
                logger.warning("Segmentation failed for %s: %s", os.path.basename(img_path), e)  
                unsupported_files.append(os.path.basename(img_path))
        
        return unsupported_files

    def run_inference(self):
        """Checks if the ml model is connected to the server, connects if not (and if possible), and runs inference on all images in eval_data_path"""
        if not self.ml_model.is_connected and not self.try_server_connection():
            message_title = "Warning"
            message_text = "Connection could not be established. Please check if the server is running and try again."
            return message_text, message_title

        try:
            # Get all images from eval_data_path
            image_list = self.fs_image_storage.search_images(self.eval_data_path)
            unsupported_files = self.fs_image_storage.get_unsupported_files(self.eval_data_path)
            
            if not image_list:
                message_text = "No images found in evaluation directory"
                message_title = "Warning"
                return message_text, message_title
            
            # Process images recursively
            import asyncio
            unsupported_files.extend(asyncio.run(self._segment_images_recursively(image_list, self.eval_data_path)))
            
            # Prepare response message
            if len(unsupported_files) > 0:
                message_text = (
                    "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
                Currently supported image file formats are: "
                    + ", ".join(settings.accepted_types)
                    + ". The files that were not supported are: "
                    + ", ".join(set(unsupported_files))
                )
                message_title = "Warning"
            else:
                message_text = "Success! Masks generated for all images"
                message_title = "Information"
        except Exception as e:
            message_text = f"An error has occured during segmentation: {str(e)}"
            message_title = "Error"
        
        return message_text, message_title

    def load_image(self, image_name=None):
        """
        Loads an image from the file system storage.

        :param str image_name: The name of the image file to load.
            If not provided, loads the currently selected image.

        :return: The loaded image.
        :rtype: numpy.ndarray

        """
        if image_name is None:
            return self.fs_image_storage.load_image(
                self.cur_selected_path, self.cur_selected_img
            )
        else:
            return self.fs_image_storage.load_image(self.cur_selected_path, image_name)

    def search_segs(self):
        """Searches in cur_selected_path for all possible segmentation files associated to cur_selected_img.
        These files should have a _seg extension to the cur_selected_img filename."""
        self.seg_filepaths = self.fs_image_storage.search_segs(
            self.cur_selected_path, self.cur_selected_img
        )

    def save_image(self, dst_directory, image_name, img):
        """Saves img array image in the dst_directory with filename cur_selected_img

        :param dst_directory: The destination directory where the image will be saved.
        :type dst_directory: str
        :param image_name: The name of the image file.
        :type image_name: str
        :param img: The image that will be saved.
        :type img: numpy.ndarray
        """
        self.fs_image_storage.save_image(dst_directory, image_name, img)

    def move_images(self, dst_directory, move_segs=False):
        """
        Moves cur_selected_img image from the current directory to the dst_directory.

        :param dst_directory: The destination directory where the images will be moved.
        :type dst_directory: str

        :param move_segs: If True, moves the corresponding segmentation along with the image. Default is False.
        :type move_segs: bool

        """
        # if image_name is None:
        self.fs_image_storage.move_image(
            self.cur_selected_path, dst_directory, self.cur_selected_img
        )
        if move_segs:
            for seg_name in self.seg_filepaths:
                self.fs_image_storage.move_image(
                    self.cur_selected_path, dst_directory, seg_name
                )

    def delete_images(self, image_names):
        """If image_name in the image_names list exists in the current directory it is deleted.

        :param image_names: A list of image names to be deleted.
        :type image_names: list[str]
        """
        for image_name in image_names:
            if os.path.exists(os.path.join(self.cur_selected_path, image_name)):
                self.fs_image_storage.delete_image(self.cur_selected_path, image_name)
