from pathlib import Path
import json
from copy import deepcopy
import numpy as np
from scipy.ndimage import find_objects
from skimage import measure
from copy import deepcopy
import SimpleITK as sitk
from radiomics import  shape2D

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


def crop_centered_padded_patch(img: np.ndarray, 
                               patch_center_xy, 
                               patch_size, 
                               obj_label,
                               mask: np.ndarray=None,
                               noise_intensity=None) -> np.ndarray:
    """
    Crop a patch from an array `x` centered at coordinates `c` with size `p`, and apply padding if necessary.

    Args:
        img (np.ndarray): The input array from which the patch will be cropped.
        patch_center_xy (tuple): The coordinates (row, column) at the center of the patch.
        patch_size (tuple): The size of the patch to be cropped (height, width).
        obj_label (int): The instance label of the mask at the patch
        mask (np.ndarray, optional): The mask array that asociated with the array x; 
                                    mask is used during training to mask out non-central elements; 
                                    for RandomForest, it is used to calculate pyradiomics features.
        noise_intensity (float, optional): Intensity of noise to be added to the background. 

    Returns:
        np.ndarray: The cropped patch with applied padding.
    """           

    height, width = patch_size  # Size of the patch
    img_height, img_width = img.shape[0], img.shape[1] # Size of the input image
    
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
        if noise_intensity is not None: img[mask_other_objs] = np.random.normal(scale=noise_intensity, size=img[mask_other_objs].shape)
        mask[mask_other_objs] = 0
        # crop the mask
        mask = mask[max(top, 0):min(bottom, img_height), max(left, 0):min(right, img_width), :]

    patch = img[max(top, 0):min(bottom, img_height), max(left, 0):min(right, img_width), :]    
    # Calculate the required padding amounts and apply padding if necessary
    if left < 0: 
        patch = np.hstack((
            np.random.normal(scale=noise_intensity, size=(patch.shape[0], abs(left), patch.shape[2])).astype(np.uint8),
            patch))
        if mask is not None: 
            mask = np.hstack((
            np.zeros((mask.shape[0], abs(left), mask.shape[2])).astype(np.uint8),
            mask))
    # Apply padding on the right side if necessary
    if right > img_width: 
        patch = np.hstack((
            patch,
            np.random.normal(scale=noise_intensity, size=(patch.shape[0], (right - img_width), patch.shape[2])).astype(np.uint8)))
        if mask is not None: 
            mask = np.hstack((
            mask,
            np.zeros((mask.shape[0], (right - img_width), mask.shape[2])).astype(np.uint8)))
    # Apply padding on the top side if necessary
    if top < 0: 
        patch = np.vstack((
            np.random.normal(scale=noise_intensity, size=(abs(top), patch.shape[1], patch.shape[2])).astype(np.uint8),
            patch))
        if mask is not None: 
            mask = np.vstack((
            np.zeros((abs(top), mask.shape[1], mask.shape[2])).astype(np.uint8),
            mask))
    # Apply padding on the bottom side if necessary
    if bottom > img_height: 
        patch = np.vstack((
            patch, 
            np.random.normal(scale=noise_intensity, size=(bottom - img_height, patch.shape[1], patch.shape[2])).astype(np.uint8)))
        if mask is not None: 
            mask = np.vstack((
            mask, 
            np.zeros((bottom - img_height, mask.shape[1], mask.shape[2])).astype(np.uint8)))
    
    return patch, mask 


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
                         mask_class=None,
                         include_mask=False):

    ''' 
    Extracts centered patches from the input image based on the centers of objects identified in the mask.

    Args:
        img (np.array): The input image.
        mask (np.array): The mask representing the objects in the image.
        p_size (int): The size of the patches to extract.
        noise_intensity (float): The intensity of noise to add to the patches.
        mask_class (int): The class represented in the patch
        include_mask (bool): Whether or not to include mask as input argument to model

    '''

    patches, patch_masks, instance_labels, class_labels  = [], [], [], []
    # if image is 2D add an additional dim for channels
    if img.ndim<3: img = img[:, :, np.newaxis]
    if mask.ndim<3: mask = mask[:, :, np.newaxis]
    # compute center of mass of objects
    centers_of_mass, instance_labels = get_center_of_mass_and_label(mask)
    # Crop patches around each center of mass
    for c, obj_label in zip(centers_of_mass, instance_labels):
        c_x, c_y = c
        patch, patch_mask = crop_centered_padded_patch(img.copy(),
                                            (c_x, c_y),
                                            (p_size, p_size),
                                            obj_label,
                                            mask=deepcopy(mask),
                                            noise_intensity=noise_intensity)
        if include_mask:
            patch_mask = 255 * (patch_mask > 0).astype(np.uint8)
            patch = np.concatenate((patch, patch_mask), axis=-1)
            
        patches.append(patch)
        patch_masks.append(patch_mask)
        if mask_class is not None:
            # get the class instance for the specific object
            instance_labels.append(obj_label)
            class_l = int(np.unique(mask_class[mask[:,:,0]==obj_label]))
            #-1 because labels from mask start from 1, we want classes to start from 0
            class_labels.append(class_l-1)
        
    return patches, patch_masks, instance_labels, class_labels

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
    
def create_patch_dataset(imgs, masks_classes, masks_instances, noise_intensity, max_patch_size, include_mask):
    '''
    Splits img and masks into patches of equal size which are centered around the cells.
    If patch_size is not given, the algorithm should first run through all images to find the max cell size, and use
    the max cell size to define the patch size. All patches and masks should then be returned
    in the same format as imgs and masks (same type, i.e. check if tensor or np.array and same 
    convention of dims, e.g.  CxHxW)
    include_mask(bool) : Flag indicating whether to include the mask along with patches. 
    '''

    if max_patch_size is None:
        max_patch_size = np.max([find_max_patch_size(mask) for mask in masks_instances])
        

    patches, patch_masks, labels = [], [], []
    for img, mask_class, mask_instance in zip(imgs,  masks_classes, masks_instances):
        # mask_instance has dimension WxH
        # mask_class has dimension WxH
        patch, patch_mask, _, label = get_centered_patches(img,
                                            mask_instance,
                                            max_patch_size, 
                                            noise_intensity=noise_intensity,
                                            mask_class=mask_class,
                                            include_mask = include_mask)
        patches.extend(patch)
        patch_masks.extend(patch_mask)
        labels.extend(label) 
    return patches, patch_masks, labels


def get_shape_features(img, mask):
    """
    Calculate shape-based radiomic features from an image within the region defined by the mask.

    Args:
    - img (np.ndarray): The input image.
    - mask (np.ndarray): The mask corresponding to the image.

    Returns:
    - np.ndarray: An array containing the calculated shape-based radiomic features, such as:
    Elongation, Sphericity, Perimeter surface.
    """

    mask = 255 * ((mask) > 0).astype(np.uint8)

    image = sitk.GetImageFromArray(img.squeeze())
    roi_mask = sitk.GetImageFromArray(mask.squeeze())

    shape_calculator = shape2D.RadiomicsShape2D(inputImage=image, inputMask=roi_mask, label=255)
    # Calculate the shape-based radiomic features
    shape_features = shape_calculator.execute()

    return np.array(list(shape_features.values()))

def extract_intensity_features(image, mask):
    """
    Extract intensity-based features from an image within the region defined by the mask.

    Args:
    - image (np.ndarray): The input image.
    - mask (np.ndarray): The mask defining the region of interest.

    Returns:
    - np.ndarray: An array containing the extracted intensity-based features:
      median intensity, mean intensity, 25th/75th percentile intensity within the masked region.
    
    """
   
    features = {}
   
    # Ensure the image and mask have the same dimensions

    if image.shape != mask.shape:
        raise ValueError("Image and mask must have the same dimensions")

    masked_image = image[(mask>0)]
    # features["min_intensity"] = np.min(masked_image)
    # features["max_intensity"] = np.max(masked_image)
    features["median_intensity"] = np.median(masked_image)
    features["mean_intensity"] = np.mean(masked_image)
    features["25th_percentile_intensity"] = np.percentile(masked_image, 25)
    features["75th_percentile_intensity"] = np.percentile(masked_image, 75)
    
    return np.array(list(features.values()))

def create_dataset_for_rf(imgs, masks):
    """
    Extract intensity-based features from an image within the region defined by the mask.

    Args:
    - imgs (List): A list of all input images.
    - mask (List): A list of all corresponding masks defining the region of interest.

    Returns:
    - List: A list of arrays containing shape and intensity-based features
        
    """
    X = []
    for img, mask in zip(imgs, masks):

        shape_features = get_shape_features(img, mask)
        intensity_features = extract_intensity_features(img, mask)
        features_list = np.concatenate((shape_features, intensity_features), axis=0)
        X.append(features_list)
      
    return X