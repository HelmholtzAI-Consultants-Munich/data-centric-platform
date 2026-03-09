from typing import Optional
import numpy as np
from skimage.transform import resize, rescale

from dcp_server.utils.processing import normalise


class FilesystemImageStorage:
    """
    Class used for image processing for model inference (prepare image for eval, prepare mask for save).
    """

    def __init__(self, data_config: dict, model_used: str) -> None:
        self.gray = bool(data_config["gray"])
        self.rescale = bool(data_config["rescale"])
        self.model_used = model_used
        self.channel_ax = None
        self.img_height = None
        self.img_width = None

    def get_image_size_properties(self, img: np.ndarray) -> None:
        """Set properties of the image size

        :param img: Image (numpy array).
        :type img: ndarray
        """

        orig_size = img.shape
        # png and jpeg will be RGB by default and 2D
        # tif can be grayscale 2D or 3D [Z, H, W]
        # image channels have already been removed in imread if self.gray=True
        # skimage.imread reads RGB or RGBA images in always with channel axis in dim=2
        
        if self.gray == False and len(orig_size) == 2:
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            print("You have set gray to False, but only two channels found in your image. Will continue assuming grayscale image.")
            self.channel_ax = None
        elif self.gray == True and len(orig_size) == 2:
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            self.channel_ax = None
        elif self.gray == True and len(orig_size) == 3: # RGB or RGBA image
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            self.channel_ax = 2
        # if we have 3 dimensions the [Z, H, W]
        elif self.gray == False and len(orig_size) == 3:
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            print(
                "Warning: 3D image stack found. We are assuming your last dimension is your channel dimension. Please cross check this."
            )
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            self.channel_ax = 2
        else:
            print("File not currently supported. See documentation for accepted types")

    def rescale_image(self, img: np.ndarray, order: int = 2) -> np.ndarray:
        """rescale image

        :param img: Image.
        :type img: ndarray
        :param order: Order of interpolation.
        :type order: int
        :return: Rescaled image.
        :rtype: ndarray
        """

        # Cellpose segmentation runs best with 512 size? TODO: check
        max_dim = max(self.img_height, self.img_width)
        rescale_factor = max_dim / 512
        return rescale(
            img, 1 / rescale_factor, order=order, channel_axis=self.channel_ax
        )

    def resize_mask(
        self, mask: np.ndarray, channel_ax: Optional[int] = None, order: int = 0
    ) -> np.ndarray:
        """resize the mask so it matches the original image size

        :param mask: Image.
        :type mask: ndarray
        :param height: Height of the image.
        :type height: int
        :param width: Width of the image.
        :type width: int
        :param order: From scikit-image - the order of the spline interpolation. Default is 0 if image.dtype is bool and 1 otherwise.
        :type order: int
        :return: Resized image.
        :rtype: ndarray
        """
        if channel_ax is not None:
            n_channel_dim = mask.shape[channel_ax]
            output_size = [self.img_height, self.img_width]
            output_size.insert(channel_ax, n_channel_dim)
        else:
            output_size = [self.img_height, self.img_width]
        return resize(mask, output_size, order=order)


    def prepare_img_for_eval(self, img: np.ndarray) -> np.ndarray:
        """Image processing for model inference.

        :param img: the image to be processed
        :type img: np.ndarray
        :return: the loaded and processed image
        :rtype: np.ndarray
        """
        # Normalise and rescale the image
        img = normalise(img)
        print('AAAAAAAAAAA', img.shape)
        self.get_image_size_properties(img)
        # Get size properties
        if self.rescale:
            img = self.rescale_image(img)
        return img

    def prepare_mask_for_save(self, mask: np.ndarray, channel_ax: int) -> np.ndarray:
        """Prepares the mask output of the model to be saved.

        :param mask: the mask
        :type mask: np.ndarray
        :param channel_ax: the channel dimension of the mask
        :type channel_ax: int
        :return: the ready to save mask
        :rtype: np.ndarray
        """
        # Resize the mask if rescaling took place before
        if self.rescale is True:
            if len(mask.shape) < 3:
                channel_ax = None
            return self.resize_mask(mask, channel_ax)
        else:
            return mask
