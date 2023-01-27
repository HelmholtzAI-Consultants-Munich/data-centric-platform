from __future__ import annotations

import time
import typing as t
from typing import TYPE_CHECKING

import numpy as np

import bentoml
from bentoml.io import Image, Text, NumpyNdarray

from pathlib import Path
from cellpose import models
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
import os


# if TYPE_CHECKING:
#     from bentoml._internal.runner.runner import RunnerMethod

#     class RunnerImpl(bentoml.Runner):
#         evaluate: RunnerMethod


class CellposeRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self):
        self.model = models.Cellpose(gpu=True, model_type="cyto")

    @bentoml.Runnable.method(batchable=False)
    def evaluate(self, img: np.ndarray, flag: int) -> np.ndarray:
        #self should be cellpose
        if flag==0: # for simplicity and testing now I just used flags, there can be a nicer solution
            mask, _, _, _ = self.model.eval(img) #this eval is the cellposes's function eval
        elif flag==1:
            mask, _, _, _ = self.model.eval(img, z_axis=0)
        else:
            pass

        return mask
    

cellpose_runner = t.cast(
    "CellposeRunner", bentoml.Runner(CellposeRunnable, name="cellpose_runner")
)

# Save the model (you can see it by typing 'bentoml models list' in the terminal)
# I'll leave it commented as I think it is not best to have it here

# bentoml.picklable_model.save_model('cellpose', cellpose_runner)

svc = bentoml.Service("cellpose_segm_test", runners=[cellpose_runner])

@svc.api(input=Text(), output=Text())  #input path to the image output message with success and the save path
async def segment_image(input_path: str):
    img = imread(os.path.join(input_path))
    orig_size = img.shape
    seg_name = Path(input_path).stem+'_seg.tiff'
    # This is copied from our local_inference, first case flag = 0, second case flag=1 for the evaluate function above
    if Path(input_path).suffix in (".jpg", ".jpeg", ".png") or (Path(input_path).suffix in (".tiff", ".tif") and len(orig_size)==2 or (len(orig_size)==3 and (orig_size[-1]==3 or orig_size[-1]==4))):
        height, width = orig_size[0], orig_size[1]
        max_dim  = max(height, width)
        rescale_factor = max_dim/512
        img = rescale(img, 1/rescale_factor)
        mask = await cellpose_runner.evaluate.async_run(img, flag=0)
        mask = resize(mask, (height, width), order=0)

        # or 3D tiff grayscale 
    elif Path(input_path).suffix in (".tiff", ".tif") and len(orig_size)==3:
        print('Warning: 3D image stack found. We are assuming your first dimension is your stack dimension. Please cross check this.')
        height, width = orig_size[1], orig_size[2]
        max_dim = max(height, width)
        rescale_factor = max_dim/512
        img = rescale(img, 1/rescale_factor, channel_axis=0)
        mask = await cellpose_runner.evaluate.async_run(img, flag=1)
        mask = resize(mask, (orig_size[0], height, width), order=0)
        
    else: 
        pass

    save_path = os.path.join(Path(input_path).parents[0], seg_name)
    imsave(save_path, mask)

    msg = "Success. Saved in " + save_path
    return msg