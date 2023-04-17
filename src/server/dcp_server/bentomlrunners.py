from typing import List
from cellpose import models
import bentoml
import numpy as np
from models import CustomCellposeModel


class CustomRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",) #TODO add here?
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, model):
        self.model = model

    @bentoml.Runnable.method(batchable=False)
    def evaluate(self, img: np.ndarray, z_axis: int) -> np.ndarray: # flag: int
        mask, _, _ = self.model.eval(img=img, z_axis=z_axis)
        return mask

    @bentoml.Runnable.method(batchable=False)
    def train(self, imgs: List[np.ndarray], masks: List[np.ndarray], **kwargs) -> str:
        save_model_path = 'mytrainedmodel' # if we want to replace existing model here set this to self.model_path 
        self.model.train(train_data = imgs, train_labels = masks, n_epochs=2, channels=[0], save_path=save_model_path, **kwargs)
        return save_model_path