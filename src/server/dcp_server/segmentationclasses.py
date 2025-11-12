import os

from dcp_server.utils import helpers
from dcp_server.utils.fsimagestorage import FilesystemImageStorage
from dcp_server.models import CustomCellpose


class GeneralSegmentation:
    """Segmentation class. Defining the main functions needed for this project and served by service - segment image and train on images."""

    def __init__(
        self, imagestorage: FilesystemImageStorage, runner, model: CustomCellpose
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
            self.model.eval_config["segmentor"][
                "channel_axis"
            ] = self.imagestorage.channel_ax
            # Evaluate the model
            mask = await self.runner.evaluate(img=img)

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

'''
class GFPProjectSegmentation(GeneralSegmentation):
    def __init__(self, imagestorage, runner):
        super().__init__(imagestorage, runner)

    async def segment_image(self, input_path, list_of_images):
        # Do extra things needed for this project...
        pass


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
