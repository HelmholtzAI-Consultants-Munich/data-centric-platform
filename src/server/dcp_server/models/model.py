from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Model(ABC):
    def __init__(self,
                 model_config: dict,
                 train_config: dict,
                 eval_config: dict,
                 model_name: str
                 ) -> None:
        
        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.model_name = model_name

    def update_configs(self,
                       train_config: dict,
                       eval_config: dict
                       ) -> None:
        """ Update the training and evaluation configurations.

        :param train_config: Dictionary containing the training configuration.
        :type train_config: dict
        :param eval_config: Dictionary containing the evaluation configuration.
        :type eval_config: dict
        """
        self.train_config = train_config
        self.eval_config = eval_config

    @abstractmethod
    def train(self, 
              imgs: List[np.array],
              masks: List[np.array]
              ) -> None:
        pass
    
    @abstractmethod
    def eval(self,
             img: np.array
             ) -> np.array:
        pass


#from segment_anything import SamPredictor, sam_model_registry
#from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
