from PyQt5.QtWidgets import  QFileIconProvider
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
import numpy as np
from skimage.measure import find_contours, label
from skimage.draw import polygon_perimeter

from pathlib import Path, PurePath
import yaml

from dcp_client.utils import settings

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()
        self.ICON_SIZE = QSize(512,512)

    def icon(self, type: 'QFileIconProvider.IconType'):
        try:
            fn = type.filePath()
        except AttributeError: return super().icon(type) # TODO handle exception differently?

        if fn.endswith(settings.accepted_types):
            a = QPixmap(self.ICON_SIZE)
            a.load(fn)
            return QIcon(a)
        else:
            return super().icon(type)

def read_config(name, config_path = 'config.yaml') -> dict:   
    """Reads the configuration file

    :param name: name of the section you want to read (e.g. 'setup','train')
    :type name: string
    :param config_path: path to the configuration file, defaults to 'config.yaml'
    :type config_path: str, optional
    :return: dictionary from the config section given by name
    :rtype: dict
    """     
    with open(config_path) as config_file:
        config_dict =  yaml.safe_load(config_file) # json.load(config_file) for .cfg file
        # Check if config file has main mandatory keys
        assert all([i in config_dict.keys() for i in ['server']])
        return config_dict[name]

def get_relative_path(filepath): return PurePath(filepath).name

def get_path_stem(filepath): return str(Path(filepath).stem)

def get_path_name(filepath): return str(Path(filepath).name)

def get_path_parent(filepath): return str(Path(filepath).parent)

def join_path(root_dir, filepath): return str(Path(root_dir, filepath))

def check_equal_arrays(array1, array2):
    return np.array_equal(array1, array2)

class Compute4Mask:

    @staticmethod
    def get_contours(instance_mask, contours_level=None):
        '''
        Find contours of objects in the instance mask.
        This function is used to identify the contours of the objects to prevent 
        the problem of the merged objects in napari window (mask).

        Parameters:
        - instance_mask (numpy.ndarray): The instance mask array.

        Returns:
        - contour_mask (numpy.ndarray): A binary mask where the contours of all objects in the instance segmentation mask are one and the rest is background.
        '''
        instance_ids = Compute4Mask.get_unique_objects(instance_mask) # get object instance labels ignoring background
        contour_mask= np.zeros_like(instance_mask)
        for instance_id in instance_ids:
            # get a binary mask only of object
            single_obj_mask = np.zeros_like(instance_mask)
            single_obj_mask[instance_mask==instance_id] = 1
            try:
                # compute contours for mask
                contours = find_contours(single_obj_mask, contours_level)
                # sometimes little dots appeas as additional contours so remove these
                if len(contours)>1: 
                    contour_sizes = [contour.shape[0] for contour in contours]
                    contour = contours[contour_sizes.index(max(contour_sizes))].astype(int)
                else: contour = contours[0]
                # and draw onto contours mask
                rr, cc = polygon_perimeter(contour[:, 0], contour[:, 1], contour_mask.shape)
                contour_mask[rr, cc] = instance_id
            except: print("Could not create contour for instance id", instance_id)
        return contour_mask
    
    @staticmethod
    def add_contour(labels_mask, instance_mask):
        '''
        Add contours of objects to the labels mask.

        Parameters:
        - labels_mask (numpy.ndarray): The class mask array without the contour pixels annotated.
        - instance_mask (numpy.ndarray): The instance mask array.

        Returns:
        - labels_mask (numpy.ndarray): The updated class mask including contours.
        '''
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            where_instances = np.where(instance_mask==instance_id)
            # get unique class ids where the object is present
            class_vals, counts = np.unique(labels_mask[where_instances], return_counts=True)
            # and take the class id which is most heavily represented
            class_id = class_vals[np.argmax(counts)]
            # make sure instance mask and class mask match
            labels_mask[np.where(instance_mask==instance_id)] = class_id
        return labels_mask

    
    @staticmethod
    def compute_new_instance_mask(labels_mask, instance_mask):
        '''
        Given an updated labels mask, update also the instance mask accordingly. So far the user can only remove an entire object in the labels mask view.
        Therefore the instance mask can only change by entirely removing an object.

        Parameters:
        - labels_mask (numpy.ndarray): The labels mask array, with changes made by the user.
        - instance_mask (numpy.ndarray): The existing instance mask, which needs to be updated.
        Returns:
        - instance_mask (numpy.ndarray): The updated instance mask.
        '''
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            unique_items_in_class_mask = list(np.unique(labels_mask[instance_mask==instance_id]))
            if len(unique_items_in_class_mask)==1 and unique_items_in_class_mask[0]==0:
                instance_mask[instance_mask==instance_id] = 0
        return instance_mask


    @staticmethod
    def compute_new_labels_mask(labels_mask, instance_mask, original_instance_mask, old_instances):
        '''
        Given the existing labels mask, the updated instance mask is used to update the labels mask.

        Parameters:
        - labels_mask (numpy.ndarray): The existing labels mask, which needs to be updated.
        - instance_mask (numpy.ndarray): The instance mask array, with changes made by the user.
        - original_instance_mask (numpy.ndarray): The instance mask array, before the changes made by the user.
        - old_instances (List): A list of the instance label ids in original_instance_mask.
        Returns:
        - new_labels_mask (numpy.ndarray): The new labels mask, with updated changes according to those the user has made in the instance mask.
        '''
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
                    old_class_id, counts = np.unique(labels_mask[where_instance_orig], return_counts=True)
                    #assert len(old_class_id)==1
                    #old_class_id = old_class_id[0]
                    # and take the class id which is most heavily represented
                    old_class_id = old_class_id[np.argmax(counts)]
                    new_labels_mask[where_instance] = old_class_id
    
        return new_labels_mask
       
    @staticmethod
    def get_unique_objects(active_mask):
        """
        Get unique objects from the active mask.
        """
        return list(np.unique(active_mask)[1:])
    
    @staticmethod
    def assert_consistent_labels(mask):
        """
        Before saving the final mask make sure the user has not mistakenly made an error during annotation,
        such that one instance id does not correspond to exactly one class id. Also checks whether for one instance id
        multiple classes exist.
        :param mask: The mask which we want to test.
        :type mask: numpy.ndarray
        :return:
            - A boolean which is True if there is more than one connected components corresponding to an instance id and Fale otherwise.
            - A boolean which is True if there is a missmatch between the instance mask and class masks (not 1-1 correspondance) and Flase otherwise.
            - A list with all the instance ids for which more than one connected component was found.
            - A list with all the instance ids for which a missmatch between class and instance masks was found.
        :rtype :
            - bool 
            - bool
            - list[int]
            - list[int] 
        """
        user_annot_error = False
        mask_mismatch_error = False
        faulty_ids_annot = []
        faulty_ids_missmatch = []
        instance_mask, class_mask = mask[0], mask[1]
        instance_ids = Compute4Mask.get_unique_objects(instance_mask)
        for instance_id in instance_ids:
            # check if there are more than one objects (connected components) with same instance_id
            if np.unique(label(instance_mask==instance_id)).shape[0] > 2: 
                user_annot_error = True
                faulty_ids_annot.append(instance_id)
            # and check if there is a mismatch between class mask and instance mask - should never happen!
            if np.unique(class_mask[np.where(instance_mask==instance_id)]).shape[0]>1:
                mask_mismatch_error = True
                faulty_ids_missmatch.append(instance_id)

        return user_annot_error, mask_mismatch_error, faulty_ids_annot, faulty_ids_missmatch