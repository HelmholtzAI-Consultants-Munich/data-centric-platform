from skimage.io import imread, imsave
import os

from app import ImageStorage

class FilesystemImageStorage(ImageStorage):

    def load_image(self, from_directory, cur_selected_img):
        # Read the selected image and read the segmentation if any:
        return imread(os.path.join(from_directory, cur_selected_img))
    
    def move_image(self, from_directory, to_directory, cur_selected_img):
        os.replace(os.path.join(from_directory, cur_selected_img), os.path.join(to_directory, cur_selected_img))

    def save_image(self, to_directory, cur_selected_img, img):
        imsave(os.path.join(to_directory, cur_selected_img), img)
    
    def delete_image(self, from_directory, cur_selected_img):
        os.remove(os.path.join(from_directory, cur_selected_img))
