import os
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rescale

from dcp_server import utils

# Import configuration
setup_config = utils.read_config('setup', config_path = 'config.cfg')

class FilesystemImageStorage():
    """Class used to deal with everything related to image storing and processing - loading, saving, transforming...
    """    
    def __init__(self, data_root, model_used):
        self.root_dir = data_root
        self.model_used = model_used
    
    def load_image(self, cur_selected_img, is_gray=True):
        """Load the image (using skiimage)

        :param cur_selected_img: full path of the image that needs to be loaded
        :type cur_selected_img: str
        :return: loaded image
        :rtype: ndarray
        """      
        try:
            return imread(os.path.join(self.root_dir , cur_selected_img), as_gray=is_gray)
        except ValueError: return None
    
    def save_image(self, to_save_path, img):
        """Save given image (using skiimage)

        :param to_save_path: full path to the directory that the image needs to be save into (use also image name in the path, eg. '/users/new_image.png')
        :type to_save_path: str
        :param img: image you wish to save
        :type img: ndarray
        """        
        imsave(os.path.join(self.root_dir, to_save_path), img)
    
    def search_images(self, directory):
        """Get a list of full paths of the images in the directory

        :param directory: path to the directory to search images in
        :type directory: str
        :return: list of image paths found in the directory (only image types that are supported - see config.cfg 'setup' section)
        :rtype: list
        """
        # Take all segmentations of the image from the current directory:
        directory = os.path.join(self.root_dir, directory)
        seg_files = [file_name for file_name in os.listdir(directory) if setup_config['seg_name_string'] in file_name]
        # Take the image files - difference between the list of all the files in the directory and the list of seg files and only file extensions currently accepted
        image_files = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if (file_name not in seg_files) and (utils.get_file_extension(file_name) in setup_config['accepted_types'])]
        return image_files
    
    def search_segs(self, cur_selected_img):
        """Returns a list of full paths of segmentations for an image

        :param cur_selected_img: full path of the image which segmentations we need to find
        :type cur_selected_img: str
        :return: list segmentation paths for the given image
        :rtype: list
        """        
        # Check the directory the image was selected from:
        img_directory = utils.get_path_parent(os.path.join(self.root_dir, cur_selected_img))
        # Take all segmentations of the image from the current directory:
        search_string = utils.get_path_stem(cur_selected_img) + setup_config['seg_name_string']
        #seg_files = [os.path.join(img_directory, file_name) for file_name in os.listdir(img_directory) if search_string in file_name]
        # TODO: check where this is used - copied the command from app's search_segs function (to fix the 1_seg and 11_seg bug)

        seg_files = [os.path.join(img_directory, file_name) for file_name in os.listdir(img_directory) if (search_string == utils.get_path_stem(file_name) or str(file_name).startswith(search_string))]

        return seg_files
    
    def get_image_seg_pairs(self, directory):
        """Get pairs of (image, image_seg)
        Used, e.g., in training to create training data-training labels pairs

        :param directory: path to the directory to search images and segmentations in
        :type directory: str
        :return: list of tuple pairs (image, image_seg)
        :rtype: list
        """        
        image_files = self.search_images(os.path.join(self.root_dir, directory))
        seg_files = []
        for image in image_files:
            seg = self.search_segs(image)
            #TODO - the search seg returns all the segs, but here we need only one, hence the seg[0]. Check if it is from training path? 
            seg_files.append(seg[0])
        return list(zip(image_files, seg_files))
            
    def get_unsupported_files(self, directory):
        """Get unsupported files found in the given directory

        :param directory: direcory path to search for files in
        :type directory: str
        :return: list of unsupported files
        :rtype: list
        """        
        return [file_name for file_name in os.listdir(os.path.join(self.root_dir, directory)) if utils.get_file_extension(file_name) not in setup_config['accepted_types']]
    
    def get_image_size_properties(self, img, file_extension):
        """Get properties of the image size

        :param img: image (numpy array)
        :type img: ndarray
        :param file_extension: file extension of the image as saved in the directory
        :type file_extension: str
        :return: size properties:
            - height
            - width
            - z_axis
        
        """        
    
        orig_size = img.shape
        # png and jpeg will be RGB by default and 2D 
        # tif can be grayscale 2D or 3D [Z, H, W]
        # image channels have already been removed in imread with is_gray=True
        if file_extension in (".jpg", ".jpeg", ".png"):
            height, width = orig_size[0], orig_size[1]
            z_axis = None
        elif file_extension in (".tiff", ".tif") and len(orig_size)==2:
            height, width = orig_size[0], orig_size[1]
            z_axis = None
        # if we have 3 dimensions the [Z, H, W]
        elif file_extension in (".tiff", ".tif") and len(orig_size)==3:
            print('Warning: 3D image stack found. We are assuming your first dimension is your stack dimension. Please cross check this.') 
            height, width = orig_size[1], orig_size[2]   
            z_axis = 0   
        else:
            print('File not currently supported. See documentation for accepted types')

        return height, width, z_axis
    
    def rescale_image(self, img, height, width, channel_ax=None, order=2):
        """rescale image

        :param img: image
        :type img: ndarray
        :param height: height of the image
        :type height: int
        :param width: width of the image
        :type width: int
        :param channel_ax: channel axis 
        :type channel_ax: int
        :return: rescaled image
        :rtype: ndarray
        """    
        if self.model_used == "UNet":
            height_pad = (height//16 + 1)*16 - height
            width_pad = (width//16 + 1)*16 - width
            return np.pad(img, ((0, height_pad),(0, width_pad)))
        else:
            # Cellpose segmentation runs best with 512 size? TODO: check
            max_dim  = max(height, width)
            rescale_factor = max_dim/512
            return rescale(img, 1/rescale_factor, order=order, channel_axis=channel_ax)
    
    def resize_mask(self, mask, height, width, channel_ax=None, order=2):
        """resize the mask so it matches the original image size

        :param mask: image
        :type mask: ndarray
        :param height: height of the image
        :type height: int
        :param width: width of the image
        :type width: int
        :param order: from scikit-image - the order of the spline interpolation, default is 0 if image.dtype is bool and 1 otherwise.
        :type order: int
        :return: resized image
        :rtype: ndarray
        """

        if self.model_used == "UNet":
            # we assume an order C, H, W
            if channel_ax is not None and channel_ax==0:
                height_pad = mask.shape[1] - height
                width_pad = mask.shape[2]- width
                return mask[:, :-height_pad, :-width_pad]
            elif channel_ax is not None and channel_ax==2:
                height_pad = mask.shape[0] - height
                width_pad = mask.shape[1]- width
                return mask[:-height_pad, :-width_pad, :]
            elif channel_ax is not None and channel_ax==1:
                height_pad = mask.shape[2] - height
                width_pad = mask.shape[0]- width
                return mask[:-width_pad, :, :-height_pad]

        else: 
            if channel_ax is not None:
                n_channel_dim = mask.shape[channel_ax]
                output_size = [height, width]
                output_size.insert(channel_ax, n_channel_dim)
            else: output_size = [height, width]
            return resize(mask, output_size, order=order)
    
    def prepare_images_and_masks_for_training(self, train_img_mask_pairs):
        """Image and mask processing for training.

        :param train_img_mask_pairs: list pairs of (image, image_seg) (as returned by get_image_seg_pairs() function)
        :type train_img_mask_pairs: list
        :return: lists of processed images and masks 
        :rtype: list, list
        """        
        imgs=[]
        masks=[]
        for img_file, mask_file in train_img_mask_pairs:
            img = self.load_image(img_file)
            mask = imread(mask_file)
            if self.model_used == "UNet":
                # Unet only accepts image sizes divisable by 16
                height_pad = (img.shape[0]//16 + 1)*16 - img.shape[0]
                width_pad = (img.shape[1]//16 + 1)*16 - img.shape[1]
                img = np.pad(img, ((0, height_pad),(0, width_pad)))
                mask = np.pad(mask, ((0, 0), (0, height_pad),(0, width_pad)))
            imgs.append(img)
            masks.append(mask)
        return imgs, masks