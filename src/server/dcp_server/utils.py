from pathlib import Path
import json
import numpy as np
from scipy.ndimage import find_objects, center_of_mass
from skimage import measure

def read_config(name, config_path = 'config.cfg') -> dict:   
    """Reads the configuration file

    :param name: name of the section you want to read (e.g. 'setup','train')
    :type name: string
    :param config_path: path to the configuration file, defaults to 'config.cfg'
    :type config_path: str, optional
    :return: dictionary from the config section given by name
    :rtype: dict
    """     
    with open(config_path) as config_file:
        config_dict = json.load(config_file)
        # Check if config file has main mandatory keys
        assert all([i in config_dict.keys() for i in ['setup', 'service', 'model', 'train', 'eval']])
        return config_dict[name]

def get_path_stem(filepath): return str(Path(filepath).stem)


def get_path_name(filepath): return str(Path(filepath).name)


def get_path_parent(filepath): return str(Path(filepath).parent)


def join_path(root_dir, filepath): return str(Path(root_dir, filepath))


def get_file_extension(file): return str(Path(file).suffix)


def crop_centered_padded_patch(x: np.ndarray, 
                               c, 
                               p, 
                               l,
                               mask: np.ndarray=None,
                               noise_intensity=None) -> np.ndarray:
    """
    Crop a patch from an array `x` centered at coordinates `c` with size `p`, and apply padding if necessary.

    Args:
        x (np.ndarray): The input array from which the patch will be cropped.
        c (tuple): The coordinates (row, column) at the center of the patch.
        p (tuple): The size of the patch to be cropped (height, width).
        l (int): The instance label of the mask at the patch

    Returns:
        np.ndarray: The cropped patch with applied padding.
    """           

    height, width = p  # Size of the patch

    # Calculate the boundaries of the patch
    top = c[0] - height // 2
    bottom = top + height
    
    left = c[1] - width // 2
    right = left + width

    # Crop the patch from the input array
    if mask is not None:
        mask_ = mask.max(-1) if len(mask.shape) >= 3 else mask
        # Zero out values in the patch where the mask is not equal to the central label
        # m = (mask_ != central_label) & (mask_ > 0)
        m = (mask_ != l) & (mask_ > 0)
        x[m] = 0
        if noise_intensity is not None:
            x[m] = np.random.normal(scale=noise_intensity, size=x[m].shape)

    patch = x[max(top, 0):min(bottom, x.shape[0]), max(left, 0):min(right, x.shape[1]), :]

    # Calculate the required padding amounts
    size_x, size_y = x.shape[1], x.shape[0]

    # Apply padding if necessary
    if left < 0: 
        patch = np.hstack((
            np.random.normal(scale=noise_intensity, size=(patch.shape[0], abs(left), patch.shape[2])).astype(np.uint8),
            patch))
    # Apply padding on the right side if necessary
    if right > size_x: 
        patch = np.hstack((
            patch,
            np.random.normal(scale=noise_intensity, size=(patch.shape[0], (right - size_x), patch.shape[2])).astype(np.uint8)))
    # Apply padding on the top side if necessary
    if top < 0: 
        patch = np.vstack((
            np.random.normal(scale=noise_intensity, size=(abs(top), patch.shape[1], patch.shape[2])).astype(np.uint8),
            patch))
    # Apply padding on the bottom side if necessary
    if bottom > size_y: 
        patch = np.vstack((
            patch, 
            np.random.normal(scale=noise_intensity, size=(bottom - size_y, patch.shape[1], patch.shape[2])).astype(np.uint8)))

    return patch 


def get_center_of_mass_and_label(mask: np.ndarray) -> np.ndarray:
    """
    Compute the centers of mass for each object in a mask.

    Args:
        mask (np.ndarray): The input mask containing labeled objects.

    Returns:
        list of tuples: A list of coordinates (row, column) representing the centers of mass for each object.
        list of ints: Holds the label for each object in the mask
    """

    # Compute the centers of mass for each labeled object in the mask
    '''
    return [(int(x[0]), int(x[1])) 
            for x in center_of_mass(mask, mask, np.arange(1, mask.max() + 1))]
    '''
    centers = []
    labels = []
    for region in measure.regionprops(mask):
        center = region.centroid
        centers.append((int(center[0]), int(center[1])))
        labels.append(region.label)
    return centers, labels
         

    
def get_centered_patches(img,
                         mask,
                         p_size: int,
                         noise_intensity=5,
                         mask_class=None):

    ''' 
    Extracts centered patches from the input image based on the centers of objects identified in the mask.

    Args:
        img: The input image.
        mask: The mask representing the objects in the image.
        p_size (int): The size of the patches to extract.
        noise_intensity: The intensity of noise to add to the patches.

    '''

    patches, instance_labels, class_labels  = [], [], []
    # if image is 2D add an additional dim for channels
    if img.ndim<3: img = img[:, :, np.newaxis]
    if mask.ndim<3: mask = mask[:, :, np.newaxis]
    # compute center of mass of objects
    centers_of_mass, instance_labels = get_center_of_mass_and_label(mask)
    # Crop patches around each center of mass
    for c, l in zip(centers_of_mass, instance_labels):
        c_x, c_y = c
        patch = crop_centered_padded_patch(img.copy(),
                                            (c_x, c_y),
                                            (p_size, p_size),
                                            l,
                                            mask=mask,
                                            noise_intensity=noise_intensity)
        patches.append(patch)
        if mask_class is not None:
            # get the class instance for the specific object
            instance_labels.append(l)
            class_l = int(np.unique(mask_class[mask[:,:,0]==l]))
            #-1 because labels from mask start from 1, we want classes to start from 0
            class_labels.append(class_l-1)
        
    return patches, instance_labels, class_labels

def get_objects(mask):
    return find_objects(mask)

def find_max_patch_size(mask):

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
    
def create_patch_dataset(imgs, masks_classes, masks_instances, noise_intensity, max_patch_size):
    '''
    Splits img and masks into patches of equal size which are centered around the cells.
    If patch_size is not given, the algorithm should first run through all images to find the max cell size, and use
    the max cell size to define the patch size. All patches and masks should then be returned
    in the same format as imgs and masks (same type, i.e. check if tensor or np.array and same 
    convention of dims, e.g.  CxHxW)
    '''

    if max_patch_size is None:
        max_patch_size = np.max([find_max_patch_size(mask) for mask in masks_instances])
        

    patches, labels = [], []
    for img, mask_class, mask_instance in zip(imgs,  masks_classes, masks_instances):
        # mask_instance has dimension WxH
        # mask_class has dimension WxH
        patch, _, label = get_centered_patches(img,
                                            mask_instance,
                                            max_patch_size, 
                                            noise_intensity=noise_intensity,
                                            mask_class=mask_class)
        patches.extend(patch)
        labels.extend(label) 
    return patches, labels