from typing import List
from cellpose import models
import bentoml
import numpy as np


class CellposeRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, cellpose_model_type = 'cyto'):
        #self.model = models.Cellpose(gpu=True, model_type="cyto")
        #self.model_path = # we need to add the model path here
        self.model = models.CellposeModel(gpu=True, model_type=cellpose_model_type)

    @bentoml.Runnable.method(batchable=False)
    def evaluate(self, img: np.ndarray, z_axis: int) -> np.ndarray: # flag: int
        mask, _, _ = self.model.eval(img, z_axis=z_axis)
        return mask

    @bentoml.Runnable.method(batchable=False)
    def train(self, imgs: List[np.ndarray], masks: List[np.ndarray]) -> str:
        save_model_path = 'mytrainedmodel' # if we want to replace existing model here set this to self.model_path 
        self.model.train(imgs, masks, n_epochs=2, channels=[0], save_path=save_model_path)
        return save_model_path