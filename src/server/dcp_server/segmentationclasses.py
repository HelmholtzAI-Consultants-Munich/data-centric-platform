import utils, settings
import os
import typing as t
import bentoml

settings.init()

class GeneralSegmentation():

    def __init__(self, imagestorage, runner):
        self.imagestorage = imagestorage
        self.runner = runner 
        

    async def segment_image(self, input_path, list_of_images):

        for img_filepath in list_of_images:
            # Load the image
            img = self.imagestorage.load_image(img_filepath)
            # Get size properties
            height, width, channel_ax = self.imagestorage.get_image_size_properties(img, utils.get_file_extension(img_filepath))
            img = self.imagestorage.rescale_image(img, height, width, channel_ax)
            
            # Evaluate the model
            mask = await self.runner.evaluate.async_run(img, z_axis=channel_ax)

            # Resize the mask
            mask = self.imagestorage.resize_image(mask, height, width, order=0)
            
            # Save segmentation
            seg_name = utils.get_path_stem(img_filepath) + settings.seg_name_string + '.tiff'
            self.imagestorage.save_image(os.path.join(input_path, seg_name), mask)


    async def train(self, imgs, masks):
        return await self.runner.train.async_run(imgs, masks)


class GFPProjectSegmentation(GeneralSegmentation):
    def __init__(self):
        super().__init__()

    async def segment_image(self, input_path, list_of_images):
        # Do extra things needed for this project...
        pass


class MitoProjectSegmentation(GeneralSegmentation):
    def __init__(self):
        super().__init__()

    async def segment_image(self, input_path, list_of_images):
        # Do extra things needed for this project...
        pass