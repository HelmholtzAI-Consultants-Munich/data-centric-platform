from PyQt5.QtWidgets import  QFileIconProvider, QStyledItemDelegate
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon, QImage
import numpy as np
from PIL import Image

from dcp_client.utils import settings

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        """Initializes the IconProvider with the default icon size."""
        super().__init__()
        self.ICON_SIZE = QSize(512, 512)

    def icon(self, type: QFileIconProvider.IconType) -> QIcon:
        """Returns the icon for the specified file type.

        :param type: The type of the file for which the icon is requested.
        :type type: QFileIconProvider.IconType
        :return: The icon for the file type.
        :rtype: QIcon
        """
        try:
            fn = type.filePath()
        except AttributeError:
            return super().icon(type)  # TODO handle exception differently?

        if fn.endswith(settings.accepted_types):
            try:
                # Load image using PIL to handle 64-bit images properly
                img = Image.open(fn)
                img_array = np.array(img)
                
                # Convert 64-bit types to 32-bit or 8-bit (Qt doesn't support 64-bit)
                if img_array.dtype == np.float64:
                    # Assume normalized (0-1) or raw values, scale to 0-255
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = np.clip(img_array / img_array.max() * 255, 0, 255).astype(np.uint8)
                elif img_array.dtype == np.int64:
                    # Convert int64 to uint8
                    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                elif img_array.dtype == np.uint16:
                    # Convert uint16 to uint8
                    img_array = (img_array / 65535.0 * 255).astype(np.uint8)
                elif img_array.dtype not in [np.uint8]:
                    # Handle any other dtype - convert to uint8
                    img_array = img_array.astype(np.uint8)
                
                # Convert to PIL Image in RGB mode
                if len(img_array.shape) == 2:
                    # Grayscale
                    pil_img = Image.fromarray(img_array, mode='L').convert('RGB')
                elif img_array.shape[2] == 1:
                    # Single channel
                    pil_img = Image.fromarray(img_array[:, :, 0], mode='L').convert('RGB')
                elif img_array.shape[2] == 3:
                    # RGB
                    pil_img = Image.fromarray(img_array, mode='RGB')
                elif img_array.shape[2] == 4:
                    # RGBA - convert to RGB
                    pil_img = Image.fromarray(img_array[:, :, :3], mode='RGB')
                else:
                    # Fallback
                    pil_img = Image.fromarray(img_array).convert('RGB')
                
                # Resize to icon size
                pil_img.thumbnail((self.ICON_SIZE.width(), self.ICON_SIZE.height()), Image.Resampling.LANCZOS)
                
                # Convert PIL image to QImage then QPixmap
                data = pil_img.tobytes("raw", "RGB")
                qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimage)
                
                return QIcon(pixmap)
            except Exception as e:
                # Fallback to default behavior if image loading fails
                return super().icon(type)
        else:
            return super().icon(type)


class CustomItemDelegate(QStyledItemDelegate):
    """
    A custom item delegate for setting a fixed height for items in a view.
    This delegate overrides the sizeHint method to set a fixed height for items.
    """
    def __init__(self, parent=None):
        """
        Initialize the CustomItemDelegate.

        :param parent: The parent QObject. Default is None.
        :type parent: QObject
        """
        super().__init__(parent)

    def sizeHint(self, option, index):
        """
        Returns the size hint for the item specified by the given index.

        :param option: The parameters used to draw the item.
        :type option: QStyleOptionViewItem
        
        :param index: The model index of the item.
        :type index: QModelIndex
        
        :returns: The size hint for the item.
        :rtype: QSize
        """
        size = super().sizeHint(option, index)
        size.setHeight(100)  
        return size
