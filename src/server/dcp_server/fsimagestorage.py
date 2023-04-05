from skimage.io import imread, imsave
from skimage.transform import resize, rescale
import os
from skimage.color import rgb2gray

import utils
import settings
settings.init()

class FilesystemImageStorage():

    def load_image(self, cur_selected_img):
        return imread(cur_selected_img)
    
    def save_image(self, to_directory, img):
        imsave(to_directory, img)

    def search_images(self, directory):
        """Returns a list of full paths of the images in the directory"""
        # Take all segmentations of the image from the current directory:
        seg_files = [file_name for file_name in os.listdir(directory) if settings.seg_name_string in file_name]
        # Take the image files - difference between the list of all the files in the directory and the list of seg files and only file extensions currently accepted
        image_files = [os.path.join(directory, file_name) for file_name in os.listdir(directory) if (file_name not in seg_files) and (utils.get_file_extension(file_name) in settings.accepted_types)]
        return image_files
    
    def search_segs(self, cur_selected_img):
        """Returns a list of full paths of segmentations for an image"""
        # Check the directory the image was selected from:
        img_directory = utils.get_path_parent(cur_selected_img)
        # Take all segmentations of the image from the current directory:
        search_string = utils.get_path_stem(cur_selected_img) + '_seg'
        seg_files = [os.path.join(img_directory, file_name) for file_name in os.listdir(img_directory) if search_string in file_name]
        return seg_files
    
    def get_unsupported_files(self, directory):
        return [file_name for file_name in os.listdir(directory) if utils.get_file_extension(file_name) not in settings.accepted_types]
    
    def get_image_size_properties(self, img, file_extension):
    
        orig_size = img.shape
        # png and jpeg will be RGB by default and 2D
        # tif can be grayscale 2D or 2D RGB and RGBA
        if file_extension in (".jpg", ".jpeg", ".png") or (file_extension in (".tiff", ".tif") and len(orig_size)==2 or (len(orig_size)==3 and (orig_size[-1]==3 or orig_size[-1]==4))):
            height, width = orig_size[0], orig_size[1]
            channel_ax = None
        # or 3D tiff grayscale 
        elif file_extension in (".tiff", ".tif") and len(orig_size)==3:
            print('Warning: 3D image stack found. We are assuming your first dimension is your stack dimension. Please cross check this.')
            height, width = orig_size[1], orig_size[2]
            channel_ax = 0                
        
        else:
            pass

        return height, width, channel_ax
    
    def rescale_image(self, img, height, width, channel_ax):
        max_dim  = max(height, width)
        rescale_factor = max_dim/512
        return rescale(img, 1/rescale_factor, channel_axis=channel_ax)
    
    def resize_image(self, img, height, width, order):
        return resize(img, (height, width), order=order)
    
    def prepare_images_and_masks_for_training(self, train_imgs, train_masks):
        '''TODO: Better name for this function? '''
        imgs=[]
        masks=[]
        for img_file, mask_file in zip(train_imgs, train_masks):
            imgs.append(rgb2gray(imread(img_file)))
            masks.append(imread(mask_file))