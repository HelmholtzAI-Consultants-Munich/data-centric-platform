import utils
import os

# Import configuration
setup_config = utils.read_config('setup', config_path = 'config.cfg')

class GeneralSegmentation():

    def __init__(self, imagestorage, runner, model):
        self.imagestorage = imagestorage
        self.runner = runner 
        self.model = model
        

    async def segment_image(self, input_path, list_of_images):

        for img_filepath in list_of_images:
            # Load the image
            img = self.imagestorage.load_image(img_filepath)
            # Get size properties
            height, width, channel_ax = self.imagestorage.get_image_size_properties(img, utils.get_file_extension(img_filepath))
            img = self.imagestorage.rescale_image(img, height, width, channel_ax)
            
            # Evaluate the model
            mask = await self.runner.evaluate.async_run(img = img, z_axis=channel_ax)

            # Resize the mask
            mask = self.imagestorage.resize_image(mask, height, width, order=0)
            
            # Save segmentation
            seg_name = utils.get_path_stem(img_filepath) + setup_config['seg_name_string'] + '.tiff'
            self.imagestorage.save_image(os.path.join(input_path, seg_name), mask)


    async def train(self, input_path):
        train_img_mask_pairs = self.imagestorage.get_image_seg_pairs(input_path)

        if not train_img_mask_pairs:
            return "No images and segs found"
                
        imgs, masks = self.imagestorage.prepare_images_and_masks_for_training(train_img_mask_pairs)

        return await self.runner.train.async_run(imgs, masks)


class GFPProjectSegmentation(GeneralSegmentation):
    def __init__(self, imagestorage, runner):
        super().__init__(imagestorage, runner)

    async def segment_image(self, input_path, list_of_images):
        # Do extra things needed for this project...
        pass


class MitoProjectSegmentation(GeneralSegmentation):
    def __init__(self, imagestorage, runner, model):
        super().__init__(imagestorage, runner, model)

    # The only difference is in segment image
    async def segment_image(self, input_path, list_of_images):

        for img_filepath in list_of_images:
            # Load the image
            img = self.imagestorage.load_image(img_filepath)
            # Get size properties
            height, width, channel_ax = self.imagestorage.get_image_size_properties(img, utils.get_file_extension(img_filepath))
            img = self.imagestorage.rescale_image(img, height, width, channel_ax)
            
            # Evaluate the model
            mask = await self.runner.evaluate.async_run(img = img, z_axis=channel_ax)

            # Resize the mask
            mask = self.imagestorage.resize_image(mask, height, width, order=0)

            # get the outlines of the segmented objects
            outlines = self.model.masks_to_outlines(mask) #[True, False] outputs
            new_mask = mask.copy()

            new_mask[mask!=0] = 2
            new_mask[outlines==True] = 1
            
            # Save segmentation
            seg_name = utils.get_path_stem(img_filepath) + setup_config['seg_name_string'] + '.tiff'
            self.imagestorage.save_image(os.path.join(input_path, seg_name), new_mask)
