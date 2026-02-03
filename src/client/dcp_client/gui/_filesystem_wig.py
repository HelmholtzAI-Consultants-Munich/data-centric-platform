import os
import numpy as np
from skimage.io import imread
from skimage.color import label2rgb


from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtCore import Qt, QVariant, QDir
from PyQt5.QtGui import QImage, QPixmap, QIcon
from functools import lru_cache

from dcp_client.utils import settings

class MyQFileSystemModel(QFileSystemModel):
    def __init__(self, app):
        """
        Initializes a custom QFileSystemModel with image caching for performance.
        """
        super().__init__()
        self.app = app
        self.img_x = 100
        self.img_y = 100
        # Cache for processed QImages to avoid reprocessing on every scroll
        self._image_cache = {}

    def setFilter(self, filters):
        """
        Sets filters for the model.

        :param filters: The filters to be applied. (QDir.Filters)
        """
        filters |= QDir.NoDotAndDotDot | QDir.AllDirs | QDir.Files
        super().setFilter(filters)

        # Exclude files containing '_seg' in their names
        self.addFilter(lambda fileInfo: "_seg" not in fileInfo.fileName())

    def addFilter(self, filterFunc):
        """
        Adds a custom filter function to the model.

        :param filterFunc: The filter function to be added. (function)
        """
        self.filterFunc = filterFunc

    def headerData(self, section, orientation, role):
        """
        Reimplemented method to provide custom header data for the model's headers.

        :param section: The section (column) index. (int)
        :param orientation: The orientation of the header. (Qt.Orientation)
        :param role: The role of the header data. (int)
        :rtype: QVariant
        """
        if section == 0 and role == Qt.DisplayRole:
            return ""
        else:
            return super().headerData(section, orientation, role)

    def _process_mask_image(self, filepath_img: str) -> QImage:
        """
        Process a mask image for display.
        
        :param filepath_img: Path to the mask image
        :return: Scaled QImage for display
        """
        try:
            img = imread(filepath_img)
            if img.ndim > 2:
                img = img[0]
            img = label2rgb(img)
            img = (255 * img.copy()).astype(np.uint8)
            height, width = img.shape[0], img.shape[1]
            qimg = QImage(img, width, height, 3 * width, QImage.Format_RGB888)
            return qimg.scaled(self.img_x, self.img_y, Qt.KeepAspectRatio)
        except Exception as e:
            return QImage()

    def _process_regular_image(self, filepath_img: str) -> QImage:
        """
        Load and process a regular image file.
        
        :param filepath_img: Path to the image file
        :return: Scaled QImage for display
        """
        try:
            qimg = QImage(filepath_img)
            return qimg.scaled(self.img_x, self.img_y, Qt.KeepAspectRatio)
        except Exception as e:
            return QImage()

    def data(self, index, role=Qt.DisplayRole):
        """
        Reimplemented method to provide custom data for the model's items.
        Uses caching to avoid reprocessing images on every scroll event.

        :param index: The index of the item. (QModelIndex)
        :param role: The role of the data. (int)
        :rtype: QVariant
        """
        if not index.isValid():
            return QVariant()
        
        if role == Qt.DecorationRole:
            filepath_img = self.filePath(index)
            
            # if an image of our dataset
            if filepath_img.endswith(settings.accepted_types):
                # Check cache first - this is the key optimization
                if filepath_img in self._image_cache:
                    return self._image_cache[filepath_img]
                
                # Process and cache the image
                if "_seg" in filepath_img and os.path.exists(filepath_img):
                    img = self._process_mask_image(filepath_img)
                else:
                    img = self._process_regular_image(filepath_img)
                
                # Cache the result
                self._image_cache[filepath_img] = img
                return img
            
        return super().data(index, role)

    def clear_cache(self) -> None:
        """Clear the image cache to free memory."""
        self._image_cache.clear()

