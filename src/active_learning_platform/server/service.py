from __future__ import annotations
import os
import time
import typing as t
from typing import List
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage.transform import resize, rescale
from cellpose import models

import bentoml
from bentoml.io import Image, Text, NumpyNdarray

accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")

# if TYPE_CHECKING:
#     from bentoml._internal.runner.runner import RunnerMethod

#     class RunnerImpl(bentoml.Runner):
#         evaluate: RunnerMethod


class CellposeRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        #self.model = models.Cellpose(gpu=True, model_type="cyto")
        #self.model_path = # we need to add the model path here
        self.model = models.CellposeModel(gpu=True, model_type="cyto")

    @bentoml.Runnable.method(batchable=False)
    def evaluate(self, img: np.ndarray, z_axis: int) -> np.ndarray: # flag: int
        #self should be cellpose
        mask, _, _ = self.model.eval(img, z_axis=z_axis)
        return mask

    @bentoml.Runnable.method(batchable=False)
    def train(self, imgs: List[np.ndarray], masks: List[np.ndarray]) -> str:
        save_model_path = 'mytrainedmodel' # if we want to replace existing model here set this to self.model_path 
        self.model.train(imgs, masks, n_epochs=2, channels=[0], save_path=save_model_path)
        return save_model_path

cellpose_runner = t.cast(
    "CellposeRunner", bentoml.Runner(CellposeRunnable, name="cellpose_runner")
)

# Save the model (you can see it by typing 'bentoml models list' in the terminal)
# I'll leave it commented as I think it is not best to have it here
# bentoml.picklable_model.save_model('cellpose', cellpose_runner)

svc = bentoml.Service("cellpose_segm_test", runners=[cellpose_runner])

@svc.api(input=Text(), output=NumpyNdarray())  #input path to the image output message with success and the save path
async def segment_image(input_path: str):

    list_files = [file for file in os.listdir(input_path) if Path(file).suffix in accepted_types]
    list_of_files_not_supported = []

    for img_filename in list_files:
        # don't do this for segmentations in the folder
        if '_seg' in img_filename:  
            continue
            #extend to check the prefix also matches an existing image
            #seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
        else:
            img = imread(os.path.join(input_path, img_filename)) #DAPI
            orig_size = img.shape
            seg_name = Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix

            # png and jpeg will be RGB by default and 2D
            # tif can be grayscale 2D or 2D RGB and RGBA
            if Path(img_filename).suffix in (".jpg", ".jpeg", ".png") or (Path(img_filename).suffix in (".tiff", ".tif") and len(orig_size)==2 or (len(orig_size)==3 and (orig_size[-1]==3 or orig_size[-1]==4))):
                height, width = orig_size[0], orig_size[1]
                channel_ax = None
            # or 3D tiff grayscale 
            elif Path(img_filename).suffix in (".tiff", ".tif") and len(orig_size)==3:
                print('Warning: 3D image stack found. We are assuming your first dimension is your stack dimension. Please cross check this.')
                height, width = orig_size[1], orig_size[2]
                channel_ax = 0                
            else: 
                list_of_files_not_supported.append(img_filename)
                continue

            max_dim  = max(height, width)
            rescale_factor = max_dim/512
            img = rescale(img, 1/rescale_factor, channel_axis=channel_ax)
            mask = await cellpose_runner.evaluate.async_run(img, z_axis=channel_ax)
            mask = resize(mask, (height, width), order=0)
            imsave(os.path.join(input_path, seg_name), mask)
            
    #msg = "Success. Segmentations saved in " + input_path
    #return msg
    return np.array(list_of_files_not_supported)

@svc.api(input=Text(), output=Text())
async def retrain(input_path):
    print("Calling retrain from server.")
    train_imgs = [os.path.join(input_path, file) for file in os.listdir(input_path) if Path(file).suffix in accepted_types and '_seg.tiff' not in file]
    train_masks = [os.path.join(input_path, file) for file in os.listdir(input_path) if '_seg.tiff' in file]
    train_imgs.sort()
    train_masks.sort()
    imgs=[]
    masks=[]
    for img_file, mask_file in zip(train_imgs, train_masks):
        imgs.append(rgb2gray(imread(img_file)))
        masks.append(imread(mask_file))
    model_path = await cellpose_runner.train.async_run(imgs, masks)
    msg = "Success! Trained model saved in: " + model_path
    return msg
    
    