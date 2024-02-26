from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Model(ABC):
    def __init__(self, model_config, train_config, eval_config, model_name):
        
        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.model_name = model_name

    @abstractmethod
    def train(self, imgs: List[np.array], masks: List[np.array]) -> None:
        pass
    
    @abstractmethod
    def eval(self, img: np.array) -> np.array:
        pass


#from segment_anything import SamPredictor, sam_model_registry
#from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
