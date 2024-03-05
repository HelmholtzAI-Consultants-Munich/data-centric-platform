from copy import deepcopy
from typing import List

import numpy as np
import torch

from dcp_server.models import CustomCellpose #  Model,
from dcp_server.models.classifiers import PatchClassifier, FeatureClassifier
from dcp_server.utils.processing import (
    get_centered_patches,
    find_max_patch_size,
    create_patch_dataset,
    create_dataset_for_rf
)

# Dictionary mapping class names to their corresponding classes

segmentor_mapping = {
    "Cellpose": CustomCellpose
}
classifier_mapping = {
    "PatchClassifier": PatchClassifier,
    "RandomForest": FeatureClassifier
}


class Inst2MultiSeg(): #Model):
    """ A two stage model for: 1. instance segmentation and 2. object wise classification  
    """
    
    def __init__(self,
                 model_name:str,
                 model_config: dict,
                 data_config: dict,
                 train_config: dict,
                 eval_config:dict
                 ) -> None:
        """ Constructs all the necessary attributes for the Inst2MultiSeg

        :param model_name: Name of the model.
        :type model_name: str
        :param model_config: Model configuration.
        :type model_config: dict
        :param data_config: Data configurations
        :type data_config: dict   
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
    """
        super().__init__()

        self.model_name = model_name
        self.model_config = model_config
        self.data_config = data_config
        self.train_config = train_config
        self.eval_config = eval_config

        self.segmentor_class = self.model_config.get("classifier").get("model_class", "Cellpose")
        self.classifier_class = self.model_config.get("classifier").get("model_class", "PatchClassifier")

        # Initialize the cellpose model and the classifier
        segmentor = segmentor_mapping.get(self.segmentor_class)
        self.segmentor = segmentor(
            self.segmentor_class, self.model_config, self.data_config, self.train_config, self.eval_config
            )
        '''
        if self.classifier_class == "PatchClassifier":
            self.classifier = PatchClassifier(
                self.classifier_class, self.model_config, self.data_config, self.train_config, self.eval_config
                )
            
        elif self.classifier_class == "RandomForest":
            self.classifier = FeatureClassifier(
                self.classifier_class, self.model_config, self.data_config, self.train_config, self.eval_config
                )
        '''
        classifier = classifier_mapping.get(self.classifier_class)
        self.classifier = classifier(
            self.classifier_class, self.model_config, self.data_config, self.train_config, self.eval_config
                )
        # make sure include mask is set to False if we are using the random forest model 
        if self.model_config["classifier"]["include_mask"] == True and self.classifier_class=="RandomForest":
            #print("Include mask=True was found, but for Random Forest, this parameter must be set to False. Doing this now.")
            self.model_config["classifier"]["include_mask"] = False
        
    def train(self,
              imgs: List[np.ndarray],
              masks: List[np.ndarray]
              ) -> None:
        """ Trains the given model. First trains the segmentor and then the clasiffier.

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray] of same shape as output of eval, i.e. one channel instances, 
        second channel classes, so [2, H, W] or [2, 3, H, W] for 3D
        """  
        # train cellpose
        masks_instances = [mask[0] for mask in masks]
        #masks_instances = list(np.array(masks)[:,0,...]) #[mask.sum(-1) for mask in masks] if masks[0].ndim == 3 else masks
        self.segmentor.train(deepcopy(imgs), masks_instances)
        masks_classes = [mask[1] for mask in masks]
        # create patch dataset to train classifier
        #masks_classes = list(
        #    masks[:,1,...]
        #) #[((mask > 0) * np.arange(1, 4)).sum(-1) for mask in masks]
        x, patch_masks, labels = create_patch_dataset(
            imgs,
            masks_classes,
            masks_instances,
            noise_intensity = self.data_config["noise_intensity"],
            max_patch_size = self.data_config["patch_size"],
            include_mask = self.model_config["classifier"]["include_mask"]
        )
        # additionally extract features from the patches if you are in RF model
        if self.classifier_class == "RandomForest": 
            x = create_dataset_for_rf(x, patch_masks)
        # train classifier
        self.classifier.train(x, labels)
        # and compute metric and loss
        self.metric = (self.segmentor.metric + self.classifier.metric) / 2
        self.loss = (self.segmentor.loss + self.classifier.loss)/2

    def eval(self,
             img: np.ndarray
             ) -> np.ndarray:
        """ Evaluate the model on the provided image and return the final mask.

        :param img: Input image for evaluation.
        :type img: np.ndarray[np.uint8]
        :return: Final mask containing instance mask and class masks.
        :rtype: np.ndarray[np.uint16]
    """
        # TBD we assume image is 2D [H, W] (see fsimage storage)
        # The final mask which is returned should have 
        # first channel the output of cellpose and the rest are the class channels
        with torch.no_grad():
            # get instance mask from segmentor
            instance_mask = self.segmentor.eval(img)
            # find coordinates of detected objects
            class_mask = np.zeros(instance_mask.shape)
            
            max_patch_size = self.data_config["patch_size"]
            if max_patch_size is None: 
                max_patch_size = find_max_patch_size(instance_mask)
            
            # get patches centered around detected objects
            x, patch_masks, instance_labels, _ = get_centered_patches(
                img,
                instance_mask,
                max_patch_size,
                noise_intensity=self.data_config["noise_intensity"],
                include_mask=self.model_config["classifier"]["include_mask"]
            )
            if self.classifier_class == "RandomForest":
                x = create_dataset_for_rf(x, patch_masks)
            # loop over patches and create classification mask
            for idx in range(len(x)):
                patch_class = self.classifier.eval(x[idx])
                # Assign predicted class to corresponding location in final_mask
                patch_class = patch_class.item() if isinstance(patch_class, torch.Tensor) else patch_class
                class_mask[instance_mask==instance_labels[idx]] = ( 
                    patch_class + 1
                )
            # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
            final_mask = np.stack(
                (instance_mask, class_mask), axis=self.eval_config['mask_channel_axis']
                ).astype(
                    np.uint16
                ) # size 2xHxW
        
        return final_mask
