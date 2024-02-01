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

def get_relative_path(filepath): return PurePath(filepath).name

def get_path_stem(filepath): return str(Path(filepath).stem)

def get_path_name(filepath): return str(Path(filepath).name)

def get_path_parent(filepath): return str(Path(filepath).parent)

def join_path(root_dir, filepath): return str(Path(root_dir, filepath))

def check_equal_arrays(array1, array2):
    return np.array_equal(array1, array2)

class Compute4Mask:

    @staticmethod
    def get_contours(instance_mask):
        '''
        Find contours of objects in the instance mask.
        This function is used to identify the contours of the objects to prevent 
        the problem of the merged objects in napari window (mask).

        Parameters:
        - instance_mask (numpy.ndarray): The instance mask array.

        Returns:
        - contour_mask (numpy.ndarray): A binary mask where the contours of all objects in the instance segmentation mask are one and the rest is background.
        '''
        labels = np.unique(instance_mask)[1:] # get object instance labels ignoring background
        contour_mask= np.zeros_like(instance_mask)
        for label in labels:
            single_obj_mask = np.zeros_like(instance_mask)
            single_obj_mask[instance_mask==label] = 1
            contours = find_contours(single_obj_mask, 0.8)
            if len(contours)>1: 
                contour_sizes = [contour.shape[0] for contour in contours]
                contour = contours[contour_sizes.index(max(contour_sizes))].astype(int)
            else: contour = contours[0]

            rr, cc = polygon_perimeter(contour[:, 0], contour[:, 1], contour_mask.shape)
            contour_mask[rr, cc] = 1
        return contour_mask
    
    @staticmethod
    def compute_new_labels_mask(labels_mask, instance_mask, original_instance_mask, old_instances):
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
                # area was added where there is background
                else:
                    old_class_id = np.unique(labels_mask[where_instance_orig])
                    #assert len(old_class_id)==1
                    old_class_id = old_class_id[0]
                    new_labels_mask[where_instance] = old_class_id
                    
        contours_mask = Compute4Mask.get_contours(instance_mask)
        new_labels_mask[contours_mask==1] = 0
        return new_labels_mask
    
    '''     
    @staticmethod
    def get_unique_objects(active_mask):
        """
        Get unique objects from the active mask.
        """

        return list(np.unique(active_mask)[1:])
    
    @staticmethod
    def find_edges(instance_mask, idx=None):
        """
        Find edges in the instance mask.
        This function is used to identify the edges of the objects to prevent 
        problem of the merged objects in napari window (mask).

        Parameters:
        - instance_mask (numpy.ndarray): The instance mask array.
        - idx (list, optional): Indices of specific labels to get contours.

        Returns:
        - numpy.ndarray: Array representing edges in the instance segmentation mask.
        """
        if idx is not None and not isinstance(idx, list):
            idx = [idx]

        instances = np.unique(instance_mask)[1:]
        edges = np.zeros_like(instance_mask).astype(int)

        if len(instances):
            for i in instances:
                if idx is None or i in idx:
        
                    mask_instance = (instance_mask == i).astype(np.uint8)

                    edge_mask = 255 * (canny(255 * (mask_instance)) > 0).astype(np.uint8)
                    edges = closing(edges, square(5))
                    edges = edges + edge_mask

            # if masks are intersecting then we want to count it only once
            edges = edges > 0
            
            return edges
        
    @staticmethod    
    def get_rounded_pos(event_position):
        """
        Get rounded position from the event position.
        """

        c, event_x, event_y = event_position
        return int(c), int(np.round(event_x)), int(np.round(event_y))
    
    @staticmethod
    def argmax (counts):
       return np.argmax(counts) if len(counts) > 0 else None
    
    @staticmethod
    def get_unique_counts_around_event(source_mask, channel_id, event_x, event_y):
        """
        Get unique counts around the specified event position in the source mask.
        """
        return np.unique(source_mask[channel_id, event_x - 1: event_x + 2, event_y - 1: event_y + 2], return_counts=True)
    
    @staticmethod
    def get_unique_counts_for_mask(source_mask, channel_id, mask_fill):
        """
        Get unique counts for the specified mask in the source mask.
        """
        return np.unique(source_mask[abs(channel_id - 1)][mask_fill], return_counts=True) 

    '''

