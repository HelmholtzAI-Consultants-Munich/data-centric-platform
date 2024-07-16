import os

import numpy as np
from skimage.color import label2rgb
from skimage.io import imread
from skimage.transform import resize

from PyQt5.QtWidgets import QFileSystemModel, QStyledItemDelegate
from PyQt5.QtCore import Qt, QRect, QVariant, QDir
from PyQt5.QtGui import QPixmap, QPainter, QImage, QBrush, QPen, QFont

class MyQFileSystemModel(QFileSystemModel):
    def __init__(self, app):
        """
        Initializes a custom QFileSystemModel
        """
        super().__init__()
        self.app = app

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

    def data(self, index, role=Qt.DisplayRole):
        """
        Reimplemented method to provide custom data for the model's items.

        :param index: The index of the item. (QModelIndex)
        :param role: The role of the data. (int)
        :rtype: QVariant
        """
        if not index.isValid():
            return QVariant()

        if "_seg" in self.filePath(index):
            return QVariant()

        if role == Qt.DisplayRole:
            filepath_img = self.filePath(index)

        if role == Qt.DecorationRole:
            filepath_img = self.filePath(index)

            if filepath_img.endswith(".tiff") or filepath_img.endswith(".png"):
                if "_seg" not in filepath_img:
                    painter = QPainter()

                    img_x, img_y = 64, 64

                    filepath_mask = f"{filepath_img.split('.')[0]}_seg.tiff"
                    img = QImage(filepath_img).scaled(img_x, img_y)

                    if os.path.exists(filepath_mask):

                        mask = imread(filepath_mask)[0]
                        num_objects = len(
                            self.app.fs_image_storage.search_segs(
                                os.path.dirname(filepath_img), filepath_img
                            )
                        )

                        mask = resize(
                            mask,
                            (int(round(1.5 * img_x)), int(round(1.5 * img_y))),
                            order=0,
                        )

                        mask = label2rgb(mask)
                        mask = (255 * np.transpose(mask, (1, 0, 2)).copy()).astype(
                            np.uint8
                        )

                        mask = QImage(
                            mask, mask.shape[1], mask.shape[0], QImage.Format_RGB888
                        ).scaled(
                            123, 123, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                        )

                        img = img.scaled(
                            82, 82, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                        )
                        img_x, img_y = 82, 82
                        painter.begin(mask)

                        rect_image = QRect(0, img_x // 2, img_x, img_y)

                        rect_corner_left = QRect(0, 0, img_x // 2, img_y // 2)
                        rect_corner_bottom = QRect(img_x, img_y, img_x // 2, img_y // 2)

                        painter.fillRect(
                            rect_corner_left, QBrush(Qt.white, Qt.SolidPattern)
                        )
                        painter.fillRect(
                            rect_corner_bottom, QBrush(Qt.white, Qt.SolidPattern)
                        )

                        painter.drawImage(rect_image, img)

                        pen = QPen(Qt.black)
                        painter.setPen(pen)

                        font = QFont()
                        font.setFamily("Arial")

                        font.setPointSize(6)
                        painter.setFont(font)

                        painter.drawText(
                            int(round((5 / 4) * img_x)) - 19,
                            int(round((5 / 4) * img_x)),
                            f"{str(num_objects)} masks",
                        )

                        painter.end()

                        pixmap = QPixmap.fromImage(mask)

                        return pixmap

                    else:
                        return img

                else:
                    return None

        return super().data(index, role)