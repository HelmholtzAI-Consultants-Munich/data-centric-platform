from copy import deepcopy
from typing import List, Optional, Union
import numpy as np

from scipy.ndimage import find_objects
from skimage import measure
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


def crop_centered_padded_patch(
    img: np.ndarray,
    patch_center_xy: tuple,
    patch_size: tuple,
    obj_label: int,
    mask: np.ndarray = None,
    noise_intensity: int = None,
) -> np.ndarray:
    """Crop a patch from an array centered at coordinates patch_center_xy with size patch_size,
    and apply padding if necessary.

    :param img: the input array from which the patch will be cropped
    :type img: np.ndarray
    :param patch_center_xy: the coordinates (row, column) at the center of the patch
    :type patch_center_xy: tuple
    :param patch_size: the size of the patch to be cropped (height, width)
    :type patch_size: tuple
    :param obj_label: the instance label of the mask at the patch
    :type obj_label: int
    :param mask: The mask array associated with the array x.
                Mask is used during training to mask out non-central elements. 
                
    :type mask: np.ndarray, optional
    :param noise_intensity: intensity of noise to be added to the background
    :type noise_intensity: float, optional
    :return: the cropped patch with applied padding
    :rtype: np.ndarray
    """

    height, width = patch_size  # Size of the patch
    img_height, img_width = img.shape[0], img.shape[1]  # Size of the input image

    # Calculate the boundaries of the patch
    top = patch_center_xy[0] - height // 2
    bottom = top + height
    left = patch_center_xy[1] - width // 2
    right = left + width

    # Crop the patch from the input array
    if mask is not None:
        mask_ = mask.max(-1) if len(mask.shape) >= 3 else mask
        # Zero out values in the patch where the mask is not equal to the central label
        mask_other_objs = (mask_ != obj_label) & (mask_ > 0)
        img[mask_other_objs] = 0
        # Add random noise at locations where other objects are present if noise_intensity is given
        if noise_intensity is not None:
            img[mask_other_objs] = np.random.normal(
                scale=noise_intensity, size=img[mask_other_objs].shape
            )
        mask[mask_other_objs] = 0
        # crop the mask
        mask = mask[
            max(top, 0) : min(bottom, img_height),
            max(left, 0) : min(right, img_width),
            :,
        ]

    patch = img[
        max(top, 0) : min(bottom, img_height), max(left, 0) : min(right, img_width), :
    ]

    # Calculate the required padding amounts and apply padding if necessary
    if left < 0:
        patch = np.hstack(
            (
                np.random.normal(
                    scale=noise_intensity,
                    size=(patch.shape[0], abs(left), patch.shape[2]),
                ).astype(np.uint8),
                patch,
            )
        )
        if mask is not None:
            mask = np.hstack(
                (
                    np.zeros((mask.shape[0], abs(left), mask.shape[2])).astype(
                        np.uint8
                    ),
                    mask,
                )
            )
    # Apply padding on the right side if necessary
    if right > img_width:
        patch = np.hstack(
            (
                patch,
                np.random.normal(
                    scale=noise_intensity,
                    size=(patch.shape[0], (right - img_width), patch.shape[2]),
                ).astype(np.uint8),
            )
        )
        if mask is not None:
            mask = np.hstack(
                (
                    mask,
                    np.zeros(
                        (mask.shape[0], (right - img_width), mask.shape[2])
                    ).astype(np.uint8),
                )
            )
    # Apply padding on the top side if necessary
    if top < 0:
        patch = np.vstack(
            (
                np.random.normal(
                    scale=noise_intensity,
                    size=(abs(top), patch.shape[1], patch.shape[2]),
                ).astype(np.uint8),
                patch,
            )
        )
        if mask is not None:
            mask = np.vstack(
                (
                    np.zeros((abs(top), mask.shape[1], mask.shape[2])).astype(np.uint8),
                    mask,
                )
            )
    # Apply padding on the bottom side if necessary
    if bottom > img_height:
        patch = np.vstack(
            (
                patch,
                np.random.normal(
                    scale=noise_intensity,
                    size=(bottom - img_height, patch.shape[1], patch.shape[2]),
                ).astype(np.uint8),
            )
        )
        if mask is not None:
            mask = np.vstack(
                (
                    mask,
                    np.zeros(
                        (bottom - img_height, mask.shape[1], mask.shape[2])
                    ).astype(np.uint8),
                )
            )
    return patch, mask


def get_center_of_mass_and_label(mask: np.ndarray) -> tuple:
    """Computes the centers of mass for each object in a mask.

    :param mask: the input mask containing labeled objects
    :type mask: np.ndarray
    :return:
        - A list of tuples representing the coordinates (row, column) of the centers of mass for each object.
        - A list of ints representing the labels for each object in the mask.
    :rtype:
        - List [tuple]
        - List [int]
    """

    # Compute the centers of mass for each labeled object in the mask

    # return [(int(x[0]), int(x[1]))
    # for x in center_of_mass(mask, mask, np.arange(1, mask.max() + 1))]

    centers = []
    labels = []
    for region in measure.regionprops(mask):
        center = region.centroid
        centers.append((int(center[0]), int(center[1])))
        labels.append(region.label)
    return centers, labels


def get_centered_patches(
    img: np.ndarray,
    mask: np.ndarray,
    p_size: int,
    noise_intensity: int = 5,
    mask_class: Optional[int] = None,
    include_mask: bool = False,
) -> tuple:
    """Extracts centered patches from the input image based on the centers of objects identified in the mask.

    :param img: The input image.
    :type img: numpy.ndarray
    :param mask: The mask representing the objects in the image.
    :type mask: numpy.ndarray
    :param p_size: The size of the patches to extract.
    :type p_size: int
    :param noise_intensity: The intensity of noise to add to the patches.
    :type noise_intensity: float
    :param mask_class: The class represented in the patch.
    :type mask_class: int
    :param include_mask: Whether or not to include the mask as an input argument to the model.
    :type include_mask: bool
    :return: A tuple containing the following elements:
            - patches (numpy.ndarray): Extracted patches.
            - patch_masks (numpy.ndarray): Masks corresponding to the extracted patches.
            - instance_labels (list): Labels identifying each object instance in the extracted patches.
            - class_labels (list): Labels identifying the class of each object instance in the extracted patches.
    :rtype: tuple
    """

    patches, patch_masks, instance_labels, class_labels = [], [], [], []
    # if image is 2D add an additional dim for channels
    if img.ndim < 3:
        img = img[:, :, np.newaxis]
    if mask.ndim < 3:
        mask = mask[:, :, np.newaxis]
    # compute center of mass of objects
    centers_of_mass, instance_labels = get_center_of_mass_and_label(mask)
    # Crop patches around each center of mass
    for c, obj_label in zip(centers_of_mass, instance_labels):
        c_x, c_y = c
        patch, patch_mask = crop_centered_padded_patch(
            img.copy(),
            (c_x, c_y),
            (p_size, p_size),
            obj_label,
            mask=deepcopy(mask),
            noise_intensity=noise_intensity,
        )
        if include_mask is True:
            patch_mask = 255 * (patch_mask > 0).astype(np.uint8)
            patch = np.concatenate((patch, patch_mask), axis=-1)

        patches.append(patch)
        patch_masks.append(patch_mask)
        if mask_class is not None:
            # get the class instance for the specific object
            instance_labels.append(obj_label)
            class_l = np.unique(mask_class[mask[:, :, 0] == obj_label])
            assert class_l.shape[0] == 1, "ERROR" + str(class_l)
            class_l = int(class_l[0])
            # -1 because labels from mask start from 1, we want classes to start from 0
            class_labels.append(class_l - 1)

    return patches, patch_masks, instance_labels, class_labels


def get_objects(mask: np.ndarray) -> List:
    """Finds labeled connected components in a binary mask.

    :param mask: The binary mask representing objects.
    :type mask: numpy.ndarray
    :return: A list of slices indicating the bounding boxes of the found objects.
    :rtype: list
    """
    return find_objects(mask)


def find_max_patch_size(mask: np.ndarray) -> float:
    """Finds the maximum patch size in a mask.

    :param mask: The binary mask representing objects.
    :type mask: numpy.ndarray
    :return: The maximum size of the bounding box edge for objects in the mask.
    :rtype: float
    """

    # Find objects in the mask
    objects = get_objects(mask)

    # Initialize variables to store the maximum patch size
    max_patch_size = 0

    # Iterate over the found objects
    for obj in objects:
        # Extract start and stop values from the slice object
        slices = [s for s in obj]
        start = [s.start for s in slices]
        stop = [s.stop for s in slices]

        # Calculate the size of the patch along each axis
        patch_size = tuple(stop[i] - start[i] for i in range(len(start)))

        # Calculate the total size (area) of the patch
        total_size = 1
        for size in patch_size:
            total_size *= size

        # Check if the current patch size is larger than the maximum
        if total_size > max_patch_size:
            max_patch_size = total_size

        max_patch_size_edge = np.ceil(np.sqrt(max_patch_size))

        return max_patch_size_edge


def create_patch_dataset(
    imgs: List[np.ndarray],
    masks_classes: Optional[Union[List[np.ndarray], torch.Tensor]],
    masks_instances: Optional[Union[List[np.ndarray], torch.Tensor]],
    noise_intensity: int,
    max_patch_size: int,
    include_mask: bool,
) -> tuple:
    """Splits images and masks into patches of equal size centered around the cells.

    :param imgs: A list of input images.
    :type imgs: list of numpy.ndarray or torch.Tensor
    :param masks_classes: A list of binary masks representing classes.
    :type masks_classes: list of numpy.ndarray or torch.Tensor
    :param masks_instances: A list of binary masks representing instances.
    :type masks_instances: list of numpy.ndarray or torch.Tensor
    :param noise_intensity: The intensity of noise to add to the patches.
    :type noise_intensity: int
    :param max_patch_size: The maximum size of the bounding box edge for objects in the mask.
    :type max_patch_size: int
    :param include_mask: A flag indicating whether to include the mask along with patches.
    :type include_mask: bool
    :return: A tuple containing the patches, patch masks, and labels.
    :rtype: tuple

    .. note::
        If patch_size is not given, the algorithm should first run through all images to find the max cell size, and use
        the max cell size to define the patch size. All patches and masks should then be returned
        in the same format as imgs and masks (same type, i.e. check if tensor or np.array and same
        convention of dims, e.g.  CxHxW)
    """
    if max_patch_size is None:
        max_patch_size = np.max([find_max_patch_size(mask) for mask in masks_instances])

    patches, patch_masks, labels = [], [], []
    for img, mask_class, mask_instance in zip(imgs, masks_classes, masks_instances):
        # mask_instance has dimension WxH
        # mask_class has dimension WxH
        patch, patch_mask, _, label = get_centered_patches(
            img=img,
            mask=mask_instance,
            p_size=max_patch_size,
            noise_intensity=noise_intensity,
            mask_class=mask_class,
            include_mask=include_mask,
        )
        patches.extend(patch)
        patch_masks.extend(patch_mask)
        labels.extend(label)
