import os
import numpy as np
from skimage.io import imread, imsave
import imageio.v3 as iio

from dcp_client.app import ImageStorage


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
