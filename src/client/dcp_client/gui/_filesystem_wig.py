import os
import numpy as np
from skimage.io import imread
from skimage.color import label2rgb


from PyQt5.QtWidgets import QFileSystemModel
from PyQt5.QtCore import Qt, QVariant, QDir, QSortFilterProxyModel
from PyQt5.QtGui import QImage, QPixmap, QIcon
from functools import lru_cache

from dcp_client.utils import settings

class SegmentationFilterProxyModel(QSortFilterProxyModel):
    """
    A proxy model that filters out segmentation files (_seg files) from the filesystem model.
    """
    
    def filterAcceptsRow(self, sourceRow, sourceParent):
        """
        Override to filter out _seg files while allowing everything else through.
        """
        # Get the source model
        source_model = self.sourceModel()
        
        # Get the index
        index = source_model.index(sourceRow, 0, sourceParent)
        
        # Get the filename
        filename = source_model.fileName(index)
        
        # Filter out _seg files
        if "_seg" in filename:
            return False
        
        # Accept everything else (including folders)
        return True


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

    def data(self, index, role=Qt.DisplayRole):
        """
        Reimplemented method to provide custom data for the model's items.
        Overlays segmentations on images when available.

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
                
                # Check if a segmentation file exists for this image
                filepath_seg = self._get_segmentation_path(filepath_img)
                
                if filepath_seg:
                    # Overlay segmentation on the image
                    img = self._overlay_segmentation_on_image(filepath_img, filepath_seg)
                else:
                    # No segmentation, just display the regular image
                    img = self._process_regular_image(filepath_img)
                
                # Cache the result
                self._image_cache[filepath_img] = img
                return img
            
        return super().data(index, role)

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

    def _get_segmentation_path(self, filepath_img: str) -> str:
        """
        Get the corresponding segmentation file path for an image.
        Handles case-insensitive extension matching and TIF/TIFF variants.
        
        :param filepath_img: Path to the image file
        :return: Path to the segmentation file, or None if it doesn't exist
        """
        # Get base name without extension
        base_path = os.path.splitext(filepath_img)[0]
        ext = os.path.splitext(filepath_img)[1]
        
        # Construct potential segmentation paths
        potential_seg_paths = [f"{base_path}_seg{ext}"]
        
        # If extension is .tif or .tiff, also check the other variant
        if ext.lower() == '.tif':
            potential_seg_paths.append(f"{base_path}_seg.tiff")
        elif ext.lower() == '.tiff':
            potential_seg_paths.append(f"{base_path}_seg.tif")
        
        # Try each potential path
        for seg_path in potential_seg_paths:
            # Check if the exact path exists
            if os.path.exists(seg_path):
                return seg_path
        
        # Try case-insensitive search in the same directory
        # This handles cases where extension case differs (e.g., .TIF vs .tif)
        directory = os.path.dirname(filepath_img)
        filename_base = os.path.basename(base_path)
        
        if directory and os.path.isdir(directory):
            try:
                for filename in os.listdir(directory):
                    filename_lower = filename.lower()
                    # Check if this file matches our segmentation pattern with various extensions
                    if (filename_lower == f"{filename_base}_seg{ext}".lower() or
                        (ext.lower() == '.tif' and filename_lower == f"{filename_base}_seg.tiff".lower()) or
                        (ext.lower() == '.tiff' and filename_lower == f"{filename_base}_seg.tif".lower())):
                        full_path = os.path.join(directory, filename)
                        if os.path.exists(full_path):
                            return full_path
            except Exception as e:
                pass
        
        return None

    def _overlay_segmentation_on_image(self, filepath_img: str, filepath_seg: str, alpha: float = 0.5) -> QImage:
        """
        Overlay a segmentation mask on an image.
        
        :param filepath_img: Path to the original image
        :param filepath_seg: Path to the segmentation mask
        :param alpha: Transparency of the overlay (0.0-1.0)
        :return: Scaled QImage with overlay for display
        """
        try:
            # Load original image
            img = imread(filepath_img)
            # Ensure it's RGB
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
            elif len(img.shape) == 3 and img.shape[2] != 3:
                # Handle other multi-channel formats
                img = img[:, :, :3]
            
            # Load and process segmentation
            seg = imread(filepath_seg)
            
            # Handle multi-channel segmentation 
            if seg.ndim > 2 and self.app.num_classes > 1:
                if len(seg.shape) == 3:
                    # (channels, height, width) format
                    seg = seg[0]
                else:
                    # Fallback: assume it's already single channel or (height, width)
                    pass
            else:
                # Single channel segmentation - extract from any multi-dimensional format
                if len(seg.shape) == 3 and seg.shape[2] > 1:
                    # (height, width, channels) format - use first channel
                    seg = seg[:, :, 0]
                elif len(seg.shape) == 3:
                    # (channels, height, width) format
                    seg = seg[0]
            
            # Convert to RGB using label2rgb
            seg_rgb = label2rgb(seg)
            seg_rgb = (255 * seg_rgb.copy()).astype(np.uint8)
            
            # Ensure images are the same size
            if img.shape[:2] != seg_rgb.shape[:2]:
                # Resize segmentation to match image
                from PIL import Image as PILImage
                seg_pil = PILImage.fromarray(seg_rgb)
                seg_pil = seg_pil.resize((img.shape[1], img.shape[0]), PILImage.Resampling.NEAREST)
                seg_rgb = np.array(seg_pil)
            
            # Ensure img is uint8
            if img.dtype != np.uint8:
                if img.dtype == np.float32 or img.dtype == np.float64:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = np.clip(img, 0, 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Blend the images
            blended = (1 - alpha) * img.astype(float) + alpha * seg_rgb.astype(float)
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # Convert to QImage
            height, width = blended.shape[0], blended.shape[1]
            qimg = QImage(blended, width, height, 3 * width, QImage.Format_RGB888)
            return qimg.scaled(self.img_x, self.img_y, Qt.KeepAspectRatio)
        except Exception as e:
            # Fallback to just the original image if overlay fails
            return self._process_regular_image(filepath_img)

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

    def clear_cache(self) -> None:
        """Clear the image cache to free memory."""
        self._image_cache.clear()
