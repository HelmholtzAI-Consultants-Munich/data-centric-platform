from __future__ import annotations
import numpy as np
import bentoml
from bentoml.io import Text, NumpyNdarray
import typing as t

from fsimagestorage import FilesystemImageStorage
from bentomlrunners import CellposeRunnable
from segmentationclasses import GeneralSegmentation

# This is where we will decide on model type, currently not working, need to figure it out
# cp = CellposeRunnable(cellpose_model_type='cyto')
custom_model_runner = t.cast(
    "CellposeRunner", bentoml.Runner(CellposeRunnable, name="cellpose_runner")
)

# This is where we decide which segmentation we use - see segmentationclasses.py
# We also decide which filesystemimagestorage we use (currently there is one, as in client)
segmentation = GeneralSegmentation(imagestorage=FilesystemImageStorage(), 
                                   runner = custom_model_runner )
                                   
svc = bentoml.Service("cellpose_segm_test", runners=[segmentation.runner])

@svc.api(input=Text(), output=NumpyNdarray())  #input path to the image output message with success and the save path
async def segment_image(input_path: str):
    
    list_of_images = segmentation.imagestorage.search_images(input_path)
    list_of_files_not_suported = segmentation.imagestorage.get_unsupported_files(input_path)

    if not list_of_images:
        return np.array(list_of_images)
    else:
        await segmentation.segment_image(input_path, list_of_images)
    
    return np.array(list_of_files_not_suported)


@svc.api(input=Text(), output=Text())
async def retrain(input_path):
    print("Calling retrain from server.")
    # Get train images (the function already checks if the files are accepted)
    train_imgs = segmentation.imagestorage.search_images(input_path).sort()
    train_masks = segmentation.imagestorage.search_segs(input_path).sort()

    # TODO: TBD: Here, we specify the _seg.tiff but now we allow for renaming it!
    # train_masks = [os.path.join(input_path, file) for file in os.listdir(input_path) if '_seg.tiff' in file]
    imgs, masks = segmentation.imagestorage.prepare_images_and_masks_for_training(train_imgs, train_masks)
    # Train the model
    model_path = segmentation.train(imgs, masks)

    msg = "Success! Trained model saved in: " + model_path

    return msg
    
    