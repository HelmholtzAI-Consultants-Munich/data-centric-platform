from PyQt5.QtWidgets import  QFileIconProvider
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
import numpy as np
from skimage.feature import canny
from skimage.morphology import closing, square
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter

from pathlib import Path, PurePath
import json

from dcp_client.utils import settings

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()
        self.ICON_SIZE = QSize(512,512)

    def icon(self, type: 'QFileIconProvider.IconType'):
        """Returns the icon for the specified file type.

        :param type: The type of the file for which the icon is requested.
        :type type: QFileIconProvider.IconType
        :return: The icon for the file type.
        :rtype: QIcon
        """
        try:
            fn = type.filePath()
        except AttributeError: return super().icon(type) # TODO handle exception differently?

        if fn.endswith(settings.accepted_types):
            a = QPixmap(self.ICON_SIZE)
            a.load(fn)
            return QIcon(a)
        else:
            return super().icon(type)

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
        assert all([i in config_dict.keys() for i in ['server']])
        return config_dict[name]

def get_relative_path(filepath):
    """Returns the name of the file from the given filepath.

    :param filepath: The path of the file.
    :type filepath: str
    :return: The name of the file.
    :rtype: str
    """
    return PurePath(filepath).name

def get_path_stem(filepath): 
    """Returns the stem (filename without its extension) from the given filepath.

    :param filepath: The path of the file.
    :type filepath: str
    :return: The stem of the file.
    :rtype: str
    """
    return str(Path(filepath).stem)

def get_path_name(filepath):
    """ Returns the name of the file from the given filepath.

    :param filepath: The path of the file.
    :type filepath: str
    :return: The name of the file.
    :rtype: str
    """
    return str(Path(filepath).name)

def get_path_parent(filepath):
    """Returns the parent directory of the given filepath.

    :param filepath: The path of the file.
    :type filepath: str
    :return: The parent directory of the file.
    :rtype: str
    """
    return str(Path(filepath).parent)

def join_path(root_dir, filepath):
    """
    Joins the root directory path with the given filepath.

    :param root_dir: The root directory.
    :type root_dir: str
    :param filepath: The path of the file.
    :type filepath: str
    :return: The joined path.
    :rtype: str
    """
    return str(Path(root_dir, filepath))

def check_equal_arrays(array1, array2):
    """Checks if two arrays are equal.

    :param array1: The first array.
    :type array1: numpy.ndarray
    :param array2: The second array.
    :type array2: numpy.ndarray
    :return: True if the arrays are equal, False otherwise.
    :rtype: bool
    """
    return np.array_equal(array1, array2)

class Compute4Mask:
    """
    Compute4Mask provides methods for manipulating masks.
    """

    @staticmethod
    def get_contours(instance_mask):
        """
        Find contours of objects in the instance mask.
        This function is used to identify the contours of 
        the objects to prevent the problem of the merged
        objects in napari window (mask).

        :param instance_mask: The instance mask array.
        :type instance_mask: numpy.ndarray
        :return: A binary mask where the contours of all
        objects in the instance segmentation mask are one
        and the rest is background.
        :rtype: numpy.ndarray
        """
        instance_ids = Compute4Mask.get_unique_objects(instance_mask) # get object instance labels ignoring background
        contour_mask= np.zeros_like(instance_mask)
        for instance_id in instance_ids:
            # get a binary mask only of object
            single_obj_mask = np.zeros_like(instance_mask)
            single_obj_mask[instance_mask==instance_id] = 1
            # compute contours for mask
            contours = find_contours(single_obj_mask, 0.8)
            # sometimes little dots appeas as additional contours so remove these
            if len(contours)>1: 
                contour_sizes = [contour.shape[0] for contour in contours]
                contour = contours[contour_sizes.index(max(contour_sizes))].astype(int)
            else: contour = contours[0]
            # and draw onto contours mask
            rr, cc = polygon_perimeter(contour[:, 0], contour[:, 1], contour_mask.shape)
            contour_mask[rr, cc] = instance_id
        return contour_mask
    
    @staticmethod
    def add_contour(labels_mask, instance_mask, contours_mask):
        """
        Add contours of objects to the labels mask.

        :param labels_mask: The class mask array without the contour
        pixels annotated.
        :type labels_mask: numpy.ndarray
        :param instance_mask: The instance mask array.
        :type instance_mask: numpy.ndarray
        :param contours_mask: The contours mask array, where each
        contour holds the instance_id.
        :type contours_mask: numpy.ndarray
        :return: The updated class mask including contours.
        :rtype: numpy.ndarray
        """
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            where_instances = np.where(instance_mask==instance_id)
            class_vals = Compute4Mask.get_unique_objects(labels_mask[where_instances])
            if len(class_vals)==0: continue
            else: 
                labels_mask[np.where(contours_mask==instance_id)] = class_vals[-1]
        return labels_mask

    
    @staticmethod
    def compute_new_instance_mask(labels_mask, instance_mask):
        """Given an updated labels mask, update also the instance mask accordingly. 
        So far the user can only remove an entire object in the labels mask view by 
        setting the color of the object to the background.
        Therefore the instance mask can only change by entirely removing an object.

        :param labels_mask: The labels mask array, with changes made by the user.
        :type labels_mask: numpy.ndarray
        :param instance_mask: The existing instance mask, which needs to be updated.
        :type instance_mask: numpy.ndarray
        :return: The updated instance mask.
        :rtype: numpy.ndarray
        """
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            unique_items_in_class_mask = list(np.unique(labels_mask[instance_mask==instance_id]))
            if len(unique_items_in_class_mask)==1 and unique_items_in_class_mask[0]==0:
                instance_mask[instance_mask==instance_id] = 0
        return instance_mask


    @staticmethod
    def compute_new_labels_mask(labels_mask, instance_mask, original_instance_mask, old_instances):
        """
        Given the existing labels mask, the updated instance mask is used to update the labels mask.

        :param labels_mask: The existing labels mask, which needs to be updated.
        :type labels_mask: numpy.ndarray
        :param instance_mask: The instance mask array, with changes made by the user.
        :type instance_mask: numpy.ndarray
        :param original_instance_mask: The instance mask array, before the changes made by the user.
        :type original_instance_mask: numpy.ndarray
        :param old_instances: A list of the instance label ids in original_instance_mask.
        :type old_instances: list
        :return: The new labels mask, with updated changes according to those the user has made in the instance mask.
        :rtype: numpy.ndarray
        """
        new_labels_mask = np.zeros_like(labels_mask)
        for instance_id in np.unique(instance_mask):
            where_instance = np.where(instance_mask==instance_id)
            # if the label is background skip
            if instance_id==0: continue
            # if the label is a newly added object, add with the same id to the labels mask
            # this is an indication to the user that this object needs to be assigned a class
            elif instance_id not in old_instances:
                new_labels_mask[where_instance] = instance_id
            else:
                where_instance_orig = np.where(original_instance_mask==instance_id)
                # if the locations of the instance haven't changed, means object wasn't changed, do nothing
                num_classes = np.unique(labels_mask[where_instance])
                # if area was erased and object retains same class
                if len(num_classes)==1: 
                    new_labels_mask[where_instance] = num_classes[0]
                # area was added where there is background or other class
                else:
                    old_class_id = np.unique(labels_mask[where_instance_orig])
                    #assert len(old_class_id)==1
                    old_class_id = old_class_id[0]
                    new_labels_mask[where_instance] = old_class_id
    
        return new_labels_mask
       
    @staticmethod
    def get_unique_objects(active_mask):
        """Gets unique objects from the active mask.

        :param active_mask: The mask array.
        :type active_mask: numpy.ndarray
        :return: A list of unique object labels.
        :rtype: list
        """
        return list(np.unique(active_mask)[1:])
    