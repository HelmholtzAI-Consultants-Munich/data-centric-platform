import os
from typing import Optional, List
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rescale

from dcp_server.utils import helpers
from dcp_server.utils.processing import pad_image, normalise

# Import configuration
setup_config = helpers.read_config("setup", config_path="config.yaml")


class FilesystemImageStorage:
    """
    Class used to deal with everything related to image storing and processing - loading, saving, transforming.
    """

    def __init__(self, data_config: dict, model_used: str) -> None:
        self.root_dir = data_config["data_root"]
        self.gray = bool(data_config["gray"])
        self.rescale = bool(data_config["rescale"])
        self.model_used = model_used
        self.channel_ax = None
        self.img_height = None
        self.img_width = None

    def load_image(
        self, cur_selected_img: str, gray: Optional[bool] = None
    ) -> Optional[np.ndarray]:
        """Load the image (using skiimage)

        :param cur_selected_img: full path of the image that needs to be loaded
        :type cur_selected_img: str
        :param gray: whether to load the image as a grayscale or not
        :type gray: bool or None, default=Nonee
        :return: loaded image
        :rtype: ndarray
        """
        if gray is None:
            gray = self.gray
        try:
            return imread(os.path.join(self.root_dir, cur_selected_img), as_gray=gray)
        except ValueError:
            return None

    def save_image(self, to_save_path: str, img: np.ndarray) -> None:
        """Save given image using skimage.

        :param to_save_path: full path to the directory that the image needs to be save into (use also image name in the path, eg. '/users/new_image.png')
        :type to_save_path: str
        :param img: image you wish to save
        :type img: ndarray
        """
        imsave(os.path.join(self.root_dir, to_save_path), img)

    def search_images(self, directory: str) -> List[str]:
        """Get a list of full paths of the images in the directory.

        :param directory: Path to the directory to search for images.
        :type directory: str
        :return: List of image paths found in the directory (only image types that are supported - see config.cfg 'setup' section).
        :rtype: list
        """
        # Take all segmentations of the image from the current directory:
        directory = os.path.join(self.root_dir, directory)
        seg_files = [
            file_name
            for file_name in os.listdir(directory)
            if setup_config["seg_name_string"] in file_name
        ]
        # Take the image files - difference between the list of all the files in the directory and the list of seg files and only file extensions currently accepted
        image_files = [
            os.path.join(directory, file_name)
            for file_name in os.listdir(directory)
            if (file_name not in seg_files)
            and (
                helpers.get_file_extension(file_name) in setup_config["accepted_types"]
            )
        ]
        return image_files

    def search_segs(self, cur_selected_img: str) -> List[str]:
        """Returns a list of full paths of segmentations for an image.

        :param cur_selected_img: Full path of the image for which segmentations are needed.
        :type cur_selected_img: str
        :return: List of segmentation paths for the given image.
        :rtype: list
        """

        # Check the directory the image was selected from:
        img_directory = helpers.get_path_parent(
            os.path.join(self.root_dir, cur_selected_img)
        )
        # Take all segmentations of the image from the current directory:
        search_string = (
            helpers.get_path_stem(cur_selected_img) + setup_config["seg_name_string"]
        )
        # seg_files = [os.path.join(img_directory, file_name) for file_name in os.listdir(img_directory) if search_string in file_name]
        # TODO: check where this is used - copied the command from app's search_segs function (to fix the 1_seg and 11_seg bug)

        seg_files = [
            os.path.join(img_directory, file_name)
            for file_name in os.listdir(img_directory)
            if (
                search_string == helpers.get_path_stem(file_name)
                or str(file_name).startswith(search_string)
            )
        ]

        return seg_files

    def get_image_seg_pairs(self, directory: str) -> List[tuple]:
        """Get pairs of (image, image_seg).

        Used, e.g., in training to create training data-training labels pairs.

        :param directory: Path to the directory to search images and segmentations in.
        :type directory: str
        :return: List of tuple pairs (image, image_seg).
        :rtype: list
        """

        image_files = self.search_images(os.path.join(self.root_dir, directory))
        seg_files = []
        for image in image_files:
            seg = self.search_segs(image)
            # TODO - the search seg returns all the segs, but here we need only one, hence the seg[0]. Check if it is from training path?
            seg_files.append(seg[0])
        return list(zip(image_files, seg_files))

    def get_unsupported_files(self, directory: str) -> List[str]:
        """Get unsupported files found in the given directory.

        :param directory: Directory path to search for files in.
        :type directory: str
        :return: List of unsupported files.
        :rtype: list
        """
        return [
            file_name
            for file_name in os.listdir(os.path.join(self.root_dir, directory))
            if not file_name.startswith(".")
            and helpers.get_file_extension(file_name)
            not in setup_config["accepted_types"]
        ]

    def get_image_size_properties(self, img: np.ndarray, file_extension: str) -> None:
        """Set properties of the image size

        :param img: Image (numpy array).
        :type img: ndarray
        :param file_extension: File extension of the image as saved in the directory.
        :type file_extension: str
        """
        # TODO simplify!

        orig_size = img.shape
        # png and jpeg will be RGB by default and 2D
        # tif can be grayscale 2D or 3D [Z, H, W]
        # image channels have already been removed in imread if self.gray=True
        # skimage.imread reads RGB or RGBA images in always with channel axis in dim=2
        if file_extension in (".jpg", ".jpeg", ".png") and self.gray == False:
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            self.channel_ax = 2
        elif file_extension in (".jpg", ".jpeg", ".png") and self.gray == True:
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            self.channel_ax = None
        elif file_extension in (".tiff", ".tif") and len(orig_size) == 2:
            self.img_height, self.img_width = orig_size[0], orig_size[1]
            self.channel_ax = None
        # if we have 3 dimensions the [Z, H, W]
        elif file_extension in (".tiff", ".tif") and len(orig_size) == 3:
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

        if self.model_used == "UNet":
            return pad_image(
                img, self.img_height, self.img_width, self.channel_ax, dividable=16
            )
        else:
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

        if self.model_used == "UNet":
            # we assume an order C, H, W
            if channel_ax is not None and channel_ax == 0:
                height_pad = mask.shape[1] - self.img_height
                width_pad = mask.shape[2] - self.img_width
                return mask[:, :-height_pad, :-width_pad]
            elif channel_ax is not None and channel_ax == 2:
                height_pad = mask.shape[0] - self.img_height
                width_pad = mask.shape[1] - self.img_width
                return mask[:-height_pad, :-width_pad, :]
            elif channel_ax is not None and channel_ax == 1:
                height_pad = mask.shape[2] - self.img_height
                width_pad = mask.shape[0] - self.img_width
                return mask[:-width_pad, :, :-height_pad]
            else:
                height_pad = mask.shape[0] - self.img_height
                width_pad = mask.shape[1] - self.img_width
                return mask[:-height_pad, :-width_pad]

        else:
            if channel_ax is not None:
                n_channel_dim = mask.shape[channel_ax]
                output_size = [self.img_height, self.img_width]
                output_size.insert(channel_ax, n_channel_dim)
            else:
                output_size = [self.img_height, self.img_width]
            return resize(mask, output_size, order=order)

    def prepare_images_and_masks_for_training(
        self, train_img_mask_pairs: List[tuple]
    ) -> tuple:
        """Image and mask processing for training.

        :param train_img_mask_pairs: List pairs of (image, image_seg) (as returned by get_image_seg_pairs() function).
        :type train_img_mask_pairs: list
        :return: Lists of processed images and masks.
        :rtype: tuple
        """

        imgs = []
        masks = []
        for img_file, mask_file in train_img_mask_pairs:
            img = self.load_image(img_file)
            img = normalise(img)
            mask = self.load_image(mask_file, gray=False)
            self.get_image_size_properties(img, helpers.get_file_extension(img_file))
            # Unet only accepts image sizes divisable by 16
            if self.model_used == "UNet":
                img = pad_image(
                    img,
                    self.img_height,
                    self.img_width,
                    channel_ax=self.channel_ax,
                    dividable=16,
                )
                mask = pad_image(
                    mask, self.img_height, self.img_width, channel_ax=0, dividable=16
                )
            if self.model_used == "CustomCellpose" and len(mask.shape) == 3:
                # if we also have class mask drop it
                mask = masks[0]  # assuming mask_channel_axis=0
            imgs.append(img)
            masks.append(mask)
        return imgs, masks

    def prepare_img_for_eval(self, img_file: str) -> np.ndarray:
        """Image processing for model inference.

        :param img_file: the path to the image
        :type img_file: str
        :return: the loaded and processed image
        :rtype: np.ndarray
        """
        # Load and normalise the image
        img = self.load_image(img_file)
        img = normalise(img)
        # Get size properties
        self.get_image_size_properties(img, helpers.get_file_extension(img_file))
        if self.rescale:
            img = self.rescale_image(img)
        return img

    def prepare_mask_for_save(self, mask: np.ndarray, channel_ax: int) -> np.ndarray:
        """Prepares the mask output of the model to be saved.

        :param mask: the mask
        :type mask: np.ndarray
        :param channel_ax: the channel dimension of the mask
        :rype channel_ax: int
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
