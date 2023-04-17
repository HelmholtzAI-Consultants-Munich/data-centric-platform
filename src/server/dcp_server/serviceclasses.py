from __future__ import annotations
import numpy as np
import bentoml
from bentoml.io import Text, NumpyNdarray


class OurBentoService():

    def __init__(self, runner, segmentation, service_name):
        self.runner = runner
        self.segmentation = segmentation
        self.service_name = service_name

    def start_service(self):

        svc = bentoml.Service(self.service_name, runners=[self.runner])

        @svc.api(input=Text(), output=NumpyNdarray())  #input path to the image output message with success and the save path
        async def segment_image(input_path: str):

            list_of_images = self.segmentation.imagestorage.search_images(input_path)
            list_of_files_not_suported = self.segmentation.imagestorage.get_unsupported_files(input_path)

            if not list_of_images:
                return np.array(list_of_images)
            else:
                await self.segmentation.segment_image(input_path, list_of_images)

            return np.array(list_of_files_not_suported)


        @svc.api(input=Text(), output=Text())
        async def retrain(input_path):
            print("Calling retrain from server.")
            # Get train images (the function already checks if the files are accepted)
            train_imgs = self.segmentation.imagestorage.search_images(input_path).sort()
            train_masks = self.segmentation.imagestorage.search_segs(input_path).sort()

            # TODO: TBD: Here, we specify the _seg.tiff but now we allow for renaming it!
            # train_masks = [os.path.join(input_path, file) for file in os.listdir(input_path) if '_seg.tiff' in file]
            imgs, masks = self.segmentation.imagestorage.prepare_images_and_masks_for_training(train_imgs, train_masks)
            # Train the model
            model_path = self.segmentation.train(imgs, masks)

            msg = "Success! Trained model saved in: " + model_path

            return msg
        
        return svc
    