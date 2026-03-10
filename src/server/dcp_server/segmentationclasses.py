import os
import numpy as np
from numpy.typing import NDArray

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

    async def segment_image(self, image: NDArray) -> NDArray:
        """Segments a single pre-loaded image.

        :param image: Pre-loaded image as numpy array
        :type image: NDArray
        :return: Segmentation mask
        :rtype: NDArray
        """
        # Prepare the image for evaluation
        prepared_img = self.imagestorage.prepare_img_for_eval(image)
        
        # Add channel ax into the model's evaluation parameters dictionary
        self.model.eval_config["segmentor"][
            "channel_axis"
        ] = self.imagestorage.channel_ax
        
        # Evaluate the model
        mask = await self.runner.evaluate(img=prepared_img)

        # And prepare the mask for saving
        mask = self.imagestorage.prepare_mask_for_save(
            mask, self.model.eval_config["mask_channel_axis"]
        )
        
        return mask