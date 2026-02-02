import os
import numpy as np
from skimage.io import imread, imsave
import imageio.v3 as iio
from typing import List

from dcp_client.app import ImageStorage
from dcp_client.utils import settings


class FilesystemImageStorage(ImageStorage):
    """FilesystemImageStorage class for handling image storage operations on the local filesystem."""

    def load_image(self, from_directory: str, cur_selected_img: str) -> np.ndarray:
        """Loads an image from the specified directory.

        :param from_directory: Path to the directory containing the image.
        :type from_directory: str
        :param cur_selected_img: Name of the image file.
        :type cur_selected_img: str
        :return: Loaded image.
        """
        filepath = os.path.join(from_directory, cur_selected_img)
        try:
            return imread(filepath)
        except Exception:
            # Fallback: use imageio which auto-detects format by magic bytes
            # This handles cases where file extension doesn't match actual format
            return np.asarray(iio.imread(filepath))

    def search_images(self, directory: str) -> List[str]:
        """Get a list of image file names in the directory (excluding segmentation files).

        :param directory: Path to the directory to search for images.
        :type directory: str
        :return: List of image file names found in the directory.
        :rtype: list
        """
        if not os.path.exists(directory):
            return []
        
        # Get all segmentation files first (files containing '_seg')
        seg_files = set()
        for file_name in os.listdir(directory):
            if settings.seg_name_string in file_name:
                seg_files.add(file_name)
        
        # Get all image files with supported extensions, excluding segmentation files
        image_files = []
        for file_name in os.listdir(directory):
            if (file_name not in seg_files and 
                any(file_name.lower().endswith(ext) for ext in settings.accepted_types)):
                image_files.append(file_name)
        
        return image_files

    def get_unsupported_files(self, directory: str) -> List[str]:
        """Get a list of files in the directory that are not supported image formats.

        :param directory: Path to the directory to search.
        :type directory: str
        :return: List of unsupported file names.
        :rtype: list
        """
        if not os.path.exists(directory):
            return []
        
        unsupported = []
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path):
                # Check if it's not a segmentation file and doesn't have supported extension
                if (settings.seg_name_string not in file_name and 
                    not any(file_name.lower().endswith(ext) for ext in settings.accepted_types)):
                    unsupported.append(file_name)
        
        return unsupported

    def move_image(self, from_directory: str, to_directory: str, cur_selected_img: str) -> None:
        """Moves an image from one directory to another.

        :param from_directory: Path to the source directory.
        :type from_directory: str
        :param to_directory: Path to the destination directory.
        :type to_directory: str
        :param cur_selected_img: Name of the image file.
        :type cur_selected_img: str
        """
        print(
            f"from:{os.path.join(from_directory, cur_selected_img)}, to:{os.path.join(to_directory, cur_selected_img)}"
        )
        os.replace(
            os.path.join(from_directory, cur_selected_img),
            os.path.join(to_directory, cur_selected_img),
        )

    def save_image(self, to_directory: str, cur_selected_img: str, img: np.ndarray) -> None:
        """Saves an image to the specified directory.

        :param to_directory: Path to the directory where the image will be saved.
        :type to_directory: str
        :param cur_selected_img: Name of the image file.
        :type cur_selected_img: str
        :param img: Image data to be saved.
        """

        imsave(os.path.join(to_directory, cur_selected_img), img)

    def delete_image(self, from_directory: str, cur_selected_img: str) -> None:
        """Deletes an image from the specified directory.

        :param from_directory: Path to the directory containing the image.
        :type from_directory: str
        :param cur_selected_img: Name of the image file.
        :type cur_selected_img: str
        """
        os.remove(os.path.join(from_directory, cur_selected_img))
