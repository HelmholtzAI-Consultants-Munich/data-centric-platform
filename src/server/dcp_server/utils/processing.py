from typing import List, Optional
import numpy as np

from scipy.ndimage import find_objects
import torch


def normalise(img: np.ndarray, norm: str = "min-max") -> np.ndarray:
    """Normalises the image based on the chosen method. Currently available methods are:
    - min max normalisation.

    :param img: image to be normalised
    :type img: np.ndarray
    :param norm: the normalisation method to apply
    :type norm: str
    :return: the normalised image
    :rtype: np.ndarray
    """
    if norm == "min-max":
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def pad_image(
    img: np.ndarray,
    height: int,
    width: int,
    channel_ax: Optional[int] = None,
    dividable: int = 16,
) -> np.ndarray:
    """Pads the image such that it is dividable by a given number.

    :param img: image to be padded
    :type img: np.ndarray
    :param height: image height
    :type height: int
    :param width: image width
    :type width: int
    :param channel_ax:
    :type channel_ax: int or None
    :param dividable: the number with which the new image size should be perfectly dividable by
    :type dividable: int
    :return: the padded image
    :rtype: np.ndarray
    """
    height_pad = (height // dividable + 1) * dividable - height
    width_pad = (width // dividable + 1) * dividable - width
    if channel_ax == 0:
        img = np.pad(img, ((0, 0), (0, height_pad), (0, width_pad)))
    elif channel_ax == 2:
        img = np.pad(img, ((0, height_pad), (0, width_pad), (0, 0)))
    else:
        img = np.pad(img, ((0, height_pad), (0, width_pad)))
    return img


def convert_to_tensor(
    imgs: List[np.ndarray], dtype: type, unsqueeze: bool = True
) -> torch.Tensor:
    """Convert the imgs to tensors of type dtype and add extra dimension if input bool is true.

    :param imgs: the list of images to convert
    :type img: List[np.ndarray]
    :param dtype: the data type to convert the image tensor
    :type dtype: type
    :param unsqueeze: If True an extra dim will be added at location zero
    :type unsqueeze: bool
    :return: the converted image
    :rtype: torch.Tensor
    """
    # Convert images tensors
    imgs = torch.stack([torch.from_numpy(img.astype(dtype)) for img in imgs])
    imgs = imgs.unsqueeze(1) if imgs.ndim == 3 and unsqueeze is True else imgs
    return imgs


def get_objects(mask: np.ndarray) -> List:
    """Finds labeled connected components in a binary mask.

    :param mask: The binary mask representing objects.
    :type mask: numpy.ndarray
    :return: A list of slices indicating the bounding boxes of the found objects.
    :rtype: list
    """
    return find_objects(mask)

