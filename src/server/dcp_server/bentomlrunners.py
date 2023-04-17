from typing import List
import bentoml
import numpy as np


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
    def train(self, imgs: List[np.ndarray], masks: List[np.ndarray]) -> str:
        self.model.train(train_data = imgs, train_labels = masks)
        return self.model.train_config['save_model_path']