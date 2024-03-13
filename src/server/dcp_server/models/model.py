from abc import ABC, abstractmethod
from typing import List
import numpy as np

class Model(ABC):
    def __init__(self,
                 model_name: str,
                 model_config: dict,
                 data_config: dict,
                 train_config: dict,
                 eval_config: dict,
                 ) -> None:
        
        self.model_name = model_name
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.eval_config = eval_config
        
        self.loss = 1e6
        self.metric = 0

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
    
    '''
    def update_configs(self,
                       config: dict,
                       ctype: str
                       ) -> None:
        """ Update the training or evaluation configurations.

        :param config: Dictionary containing the updated configuration.
        :type config: dict
        :param ctype:type of config to update, will be train or eval
        :type ctype: str
        """
        if ctype=='train': self.train_config = config
        else: self.eval_config = config
    '''


#from segment_anything import SamPredictor, sam_model_registry
#from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
