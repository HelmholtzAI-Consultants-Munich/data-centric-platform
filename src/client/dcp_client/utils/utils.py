from PyQt5.QtWidgets import  QFileIconProvider, QMessageBox
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
import numpy as np
from skimage.feature import canny, peak_local_max
from skimage.morphology import closing, square

from pathlib import Path, PurePath
import json

from dcp_client.utils import settings

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()
        self.ICON_SIZE = QSize(512,512)

    def icon(self, type: 'QFileIconProvider.IconType'):

        fn = type.filePath()

        if fn.endswith(settings.accepted_types):
            a = QPixmap(self.ICON_SIZE)
            a.load(fn)
            return QIcon(a)
        else:
            return super().icon(type)

def create_warning_box(message_text, message_title="Warning"):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText(message_text)
    msg.setWindowTitle(message_title)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()

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

class Compute4Mask:

    @staticmethod
    def get_unique_objects(active_mask):
        """
        Get unique objects from the active mask.
        """

        return set(np.unique(active_mask)[1:])
    
    @staticmethod
    def find_edges(instance_mask, idx=None):
        '''
        Find edges in the instance mask.

        Parameters:
        - instance_mask (numpy.ndarray): The instance mask array.
        - idx (list, optional): Indices of specific labels to get contours.

        Returns:
        - numpy.ndarray: Array representing edges in the instance segmentation mask.
        '''
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
       
       return np.argmax(counts)
    
    @staticmethod
    def get_unique_counts_around_event(source_mask, c, event_x, event_y):
        """
        Get unique counts around the specified event position in the source mask.
        """
        return np.unique(source_mask[c, event_x - 1: event_x + 2, event_y - 1: event_y + 2], return_counts=True)
    
    @staticmethod
    def get_unique_counts_for_mask(source_mask, c, mask_fill):
        """
        Get unique counts for the specified mask in the source mask.
        """
        return np.unique(source_mask[abs(c - 1)][mask_fill], return_counts=True) 


