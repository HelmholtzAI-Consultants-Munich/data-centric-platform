from copy import deepcopy
from typing import List
import numpy as np

import torch
from torch import nn

from cellpose import models, utils
from cellpose.metrics import aggregated_jaccard_index
from cellpose.dynamics import labels_to_flows

from .model import Model

class CustomCellpose(models.CellposeModel, Model):
    """Custom cellpose model inheriting the attributes and functions from the original CellposeModel and implementing
    additional attributes and methods needed for this project.
    """    
    def __init__(self,
                 model_name: str,
                 model_config: dict,
                 data_config: dict,
                 train_config: dict,
                 eval_config: dict,
                 ) -> None:
        """Constructs all the necessary attributes for the CustomCellpose. 
        The model inherits all attributes from the parent class, the init allows to pass any other argument that the parent class accepts.
        Please, visit here https://cellpose.readthedocs.io/en/latest/api.html#id4 for more details on arguments accepted. 

        :param model_name: The name of the current model
        :type model_name: str
        :param model_config: dictionary passed from the config file with all the arguments for the __init__ function and model initialization
        :type model_config: dict
        :param data_config: dictionary passed from the config file with all the data configurations
        :type data_config: dict        
        :param train_config: dictionary passed from the config file with all the arguments for training function
        :type train_config: dict
        :param eval_config: dictionary passed from the config file with all the arguments for eval function
        :type eval_config: dict
        """
        
        # Initialize the cellpose model
        # super().__init__(**model_config["segmentor"])
        Model.__init__(self, model_name, model_config, data_config, train_config, eval_config)
        models.CellposeModel.__init__(self, **model_config["segmentor"])
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.model_name = model_name
        self.mkldnn = False # otherwise we get error with saving model
        self.loss = 1e6
        self.metric = 0

    def train(self,
              imgs: List[np.ndarray],
              masks: List[np.ndarray]
              ) -> None:
        """Trains the given model
        Calls the original train function.

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray]
        """  
        if self.train_config["segmentor"]["n_epochs"]==0: return
        super().train(
            train_data=deepcopy(imgs), #Cellpose changes the images
            train_labels=masks,
            **self.train_config["segmentor"]
            )
        pred_masks, pred_flows, true_flows = self.compute_masks_flows(imgs, masks)
        # get loss, combination of mse for flows and bce for cell probability
        self.loss = self.loss_fn(true_flows, pred_flows) 
        self.metric = np.mean(aggregated_jaccard_index(masks, pred_masks))

    def eval(self,
             img: np.ndarray
             ) -> np.ndarray:
        """Evaluate the model - find mask of the given image
        Calls the original eval function. 

        :param img: image to evaluate on
        :type img: np.ndarray
        :return: mask of the image, list of 2D arrays, or single 3D array (if do_3D=True) labelled image.
        :rtype: np.ndarray
        """  
        # 0 to take only mask - inline with other models eval should always return the final mask
        return super().eval(x=img, **self.eval_config["segmentor"])[
            0
        ] 

    def eval_all_outputs(self,
                         img: np.ndarray
                         ) -> tuple:
        """Get all outputs of the model when running eval.

        :param img: Input image for segmentation.
        :type img: numpy.ndarray
        :return: mask, flows, styles etc. Returns the same as cellpose.models.CellposeModel.eval - see Cellpose API Guide for more details. 
        :rtype: tuple
        """

        return super().eval(x=img, **self.eval_config["segmentor"])
    
    # I introduced typing here as suggest by the docstring
    def compute_masks_flows(self, imgs: List[np.ndarray], masks:List[np.ndarray]) -> tuple:
        """ Computes instance, binary mask and flows in x and y - needed for loss and metric computations

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray]
        :return: A tuple containing the following elements:
            - pred_masks List [np.ndarray]: A list of predicted instance masks
            - pred_flows (torch.Tensor): A tensor holding the stacked predicted cell probability map, horizontal and vertical flows for all images
            - true_lbl (np.ndarray): A numpy array holding the stacked true binary mask, horizontal and vertical flows for all images 
        :rtype: tuple
        """  
        # compute for loss and metric
        true_bin_masks = [mask>0 for mask in masks] # get binary masks
        true_flows = labels_to_flows(masks) # get cellpose flows
        # get predicted flows and cell probability
        pred_masks = []
        pred_flows = []
        true_lbl = []
        for idx, img in enumerate(imgs):
            mask, flows, _ = super().eval(x=img, **self.eval_config["segmentor"])
            pred_masks.append(mask)
            pred_flows.append(np.stack([flows[1][0], flows[1][1], flows[2]])) # stack cell probability map, horizontal and vertical flow
            true_lbl.append(np.stack([true_bin_masks[idx], true_flows[idx][2], true_flows[idx][3]]))
        
        true_lbl = np.stack(true_lbl)
        pred_flows=np.stack(pred_flows)
        pred_flows = torch.from_numpy(pred_flows).float().to('cpu')
        return pred_masks, pred_flows, true_lbl

    def masks_to_outlines(self,
                          mask: np.ndarray
                          ) -> np.ndarray:
        """ get outlines of masks as a 0-1 array
        Calls the original cellpose.utils.masks_to_outlines function

        :param mask: int, 2D or 3D array, mask of an image
        :type mask: ndarray
        :return: outlines
        :rtype: ndarray
        """        
        return utils.masks_to_outlines(mask) # [True, False] outputs
