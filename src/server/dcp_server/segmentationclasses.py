import os

from dcp_server.utils import helpers
from dcp_server.utils.fsimagestorage import FilesystemImageStorage
from dcp_server import models as DCPModels


class GeneralSegmentation:
    """Segmentation class. Defining the main functions needed for this project and served by service - segment image and train on images."""

    def __init__(
        self, imagestorage: FilesystemImageStorage, runner, model: DCPModels
    ) -> None:
        """Constructs all the necessary attributes for the GeneralSegmentation.

        :param imagestorage: imagestorage system used (see fsimagestorage.py)
        :type imagestorage: FilesystemImageStorage class object
        :param runner: runner used in the service
        :type runner: CustomRunnable class object
        :param model: model used for segmentation
        :type model: class object from the models.py
        """
        self.imagestorage = imagestorage
        self.runner = runner
        self.model = model
        self.no_files_msg = "No image-label pairs found in curated directory"

    async def segment_image(self, input_path: str, list_of_images: str) -> None:
        """Segments images from the given  directory

        :param input_path: directory where the images are saved and where segmentation results will be saved
        :type input_path: str
        :param list_of_images: list of image objects from the directory that are currently supported
        :type list_of_images: list
        """

        for img_filepath in list_of_images:
            img = self.imagestorage.prepare_img_for_eval(img_filepath)
            # Add channel ax into the model's evaluation parameters dictionary
            if self.imagestorage.model_used != "UNet":
                self.model.eval_config["segmentor"][
                    "channel_axis"
                ] = self.imagestorage.channel_ax
            # Evaluate the model
            mask = await self.runner.evaluate.async_run(img=img)
            # And prepare the mask for saving
            mask = self.imagestorage.prepare_mask_for_save(
                mask, self.model.eval_config["mask_channel_axis"]
            )
            # Save segmentation
            seg_name = (
                helpers.get_path_stem(img_filepath)
                + self.imagestorage.seg_name_string
                + ".tiff"
            )
            self.imagestorage.save_image(os.path.join(input_path, seg_name), mask)

    async def train(self, input_path: str) -> str:
        """Train model on images and masks in the given input directory.
        Calls the runner's train function.

        :param input_path: directory where the images are saved
        :type input_path: str
        :return: runner's train function output - path of the saved model
        :rtype: str
        """

        train_img_mask_pairs = self.imagestorage.get_image_seg_pairs(input_path)

        if not train_img_mask_pairs:
            return self.no_files_msg

        imgs, masks = self.imagestorage.prepare_images_and_masks_for_training(
            train_img_mask_pairs
        )
        model_save_path = await self.runner.train.async_run(imgs, masks)

        return model_save_path


'''

class GFPProjectSegmentation(GeneralSegmentation):
    def __init__(self, imagestorage, runner):
        super().__init__(imagestorage, runner)
        # Apply threshold on GFP channel
        threshold_gfp = 300 # Define the threshold looking at the plots

    async def segment_image(self, input_path, list_of_images):
        # Do extra things needed for this project...

        list_files_dapi = [file for file in os.listdir(input_path) if Path(file).suffix in accepted_types and 'DAPI Channel' in file]
        list_files_gfp = [file for file in os.listdir(input_path) if Path(file).suffix in accepted_types and 'GFP Channel' in file]
        list_files_dapi.sort()
        list_files_gfp.sort()

        for idx, img_filename in enumerate(list_files_dapi):

            if '_seg' in img_filename or '_classes' in img_filename:  continue

            else:
                img = imread(os.path.join(eval_data_path, img_filename)) #DAPI
                img_gfp = imread(os.path.join(eval_data_path, list_files_gfp[idx]))
                orig_size = img.shape
                # get root filename without DAPI Channel ending
                name_split = img_filename.split('DAPI Channel')[0]
                seg_name = name_split + '_seg.tiff' #Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix
                class_name = name_split + '_classes.tiff' #Path(img_filename).stem+'_classes.tiff' #+Path(img_filename).suffix

                height, width = orig_size[1], orig_size[2]

                max_dim = max(height, width)
                rescale_factor = max_dim/512
                img = rescale(img, 1/rescale_factor, channel_axis=0)
                mask, _, _, _ = model.eval(img, z_axis=0) #for 3D
                mask = merge_2d_labels(mask)
                mask = resize(mask, (orig_size[0], height, width), order=0)

                # get labels of object segmentation
                labels = np.unique(mask)[1:]
                class_mask = np.copy(mask)
                class_mask[mask != 0] = 1 # set all objects to class 1

                # if the mean of an object in the gfp image is abpve the predefined threshold set it to class 2
                for l in labels:
                    mean_l = np.mean(img_gfp[mask == l])
                    if mean_l > threshold_gfp:
                        class_mask[mask == l] = 2                   

                imsave(os.path.join(eval_data_path, seg_name), mask)
                imsave(os.path.join(eval_data_path, class_name), class_mask)
    
    def merge_2d_labels(mask):
        inc = 100 
        for idx in range(mask.shape[0]-1):
            slice_cur = mask[idx] # get current slice
            labels_cur = list(np.unique(slice_cur))[1:] # and a list of the object labels in this slice

            mask[idx+1] += inc # increase the label values of the next slices so that we dont have different objects with same labels
            mask[idx+1][mask[idx+1]==inc] = 0 # make sure that background remains black
            inc += 100 # and increase inc by 100 so in the next slice there is no overlap again

            # for each label in the current slices
            for label_cur in labels_cur:
                # get a patch around the object
                x, y = np.where(slice_cur==label_cur)
                max_x, min_x = np.max(x), np.min(x)
                max_y, min_y = np.max(y), np.min(y)
                # and extract this patch in the next slice
                slice_patch_next = mask[idx+1, min_x:max_x, min_y:max_y]
                # get the labels within this patch
                labels_next = np.unique(slice_patch_next)

                # if there are none continue
                if labels_next.shape[0]==0: continue
                elif labels_next[0]==0 and labels_next.shape[0]==1: continue
                # if there is only one label(not the background) set it to value of label in current slice
                elif labels_next[0]!=0 and labels_next.shape[0]==1: 
                    slice_next = mask[idx+1]
                    slice_next[slice_next==labels_next[0]] = label_cur
                    mask[idx+1] = slice_next
                    continue
                # and if there are multiple
                if labels_next[0]==0 and labels_next.shape[0]>1: 
                    labels_next = labels_next[1:]
                # pick the object with the largest area 
                obj_sizes = [np.where(slice_patch_next==l2)[0].shape[0] for l2 in labels_next]
                idx_max = obj_sizes.index(max(obj_sizes))
                # replace the found label in the next slice with the one in the current slice
                label2replace = labels_next[idx_max]
                slice_next = mask[idx+1]
                slice_next[slice_next==label2replace] = label_cur
                mask[idx+1] = slice_next

        # reorder the labels in the mask so they are continuous
        unique = list(np.unique(mask))[1:] 
        for idx, l in enumerate(unique):
            mask[mask==l] = idx+1
        # and convert to unit8 if we have fewer than 256 labels
        if len(unique)<256:
            mask = mask.astype(np.uint8)
        return mask


class MitoProjectSegmentation(GeneralSegmentation):
    """ Segmentation class inheriting the attributes and functions from the original GeneralSegmentation and implementing
    additional attributes and methods needed for this project.
    """    
    def __init__(self, imagestorage, runner, model):
        """ Constructs all the necessary attributes for the MitoProjectSegmentation. Inherits all from the GeneralSegmentation

        :param imagestorage: imagestorage system used (see fsimagestorage.py)
        :type imagestorage: FilesystemImageStorage class object
        :param runner: runner used in the service
        :type runner: CustomRunnable class object
        :param model: model used for segmentation 
        :type model: class object from the models.py
        """    
        super().__init__(imagestorage, runner, model)

    # The only difference is in segment image
    async def segment_image(self, input_path, list_of_images):
        """ Segments images from the given  directory. 
        The function differs from the parent class' function in obtaining the outlines of the masks.

        :param input_path: directory where the images are saved
        :type input_path: str
        :param list_of_images: list of image objects from the directory that are currently supported
        :type list_of_images: list
        """ 

        for img_filepath in list_of_images:
            # Load the image
            img = self.imagestorage.load_image(img_filepath)
            # Get size properties
            height, width, channel_ax = self.imagestorage.get_image_size_properties(img, helpers.get_file_extension(img_filepath))
            img = self.imagestorage.rescale_image(img, height, width, channel_ax)
            
            # Add channel ax into the model's evaluation parameters dictionary
            self.model.eval_config['z_axis'] = channel_ax

            # Evaluate the model
            mask = await self.runner.evaluate.async_run(img = img, **self.model.eval_config)

            # Resize the mask
            mask = self.imagestorage.resize_mask(mask, height, width, order=0)

            # get the outlines of the segmented objects
            outlines = self.model.masks_to_outlines(mask) #[True, False] outputs
            new_mask = mask.copy()

            new_mask[mask!=0] = 2
            new_mask[outlines==True] = 1
            
            # Save segmentation
            seg_name = helpers.get_path_stem(img_filepath) + setup_config['seg_name_string'] + '.tiff'
            self.imagestorage.save_image(os.path.join(input_path, seg_name), new_mask)
        '''
