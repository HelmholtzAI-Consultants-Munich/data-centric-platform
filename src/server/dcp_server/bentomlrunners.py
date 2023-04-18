from typing import List
import bentoml
import numpy as np


class CustomRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",) #TODO add here?
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, model, save_model_path):
        
        self.model = model
        self.save_model_path = save_model_path

    @bentoml.Runnable.method(batchable=False)
    def evaluate(self, img: np.ndarray, z_axis: int) -> np.ndarray: # flag: int

        mask, _, _ = self.model.eval(img=img, z_axis=z_axis)

        return mask

    @bentoml.Runnable.method(batchable=False)
    def train(self, imgs: List[np.ndarray], masks: List[np.ndarray]) -> str:

        self.model.train(imgs, masks)
        # Save the benotml model
        bentoml.picklable_model.save_model(self.save_model_path, self.model) 

        return self.save_model_path