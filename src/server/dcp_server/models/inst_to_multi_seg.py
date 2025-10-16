from typing import List

import numpy as np
import torch

from .model import Model
from .custom_cellpose import CustomCellpose
from dcp_server.models.classifiers import PatchClassifier, FeatureClassifier
from dcp_server.utils.processing import (
    get_centered_patches,
    find_max_patch_size,
    create_patch_dataset,
    create_dataset_for_rf,
)

# Dictionary mapping class names to their corresponding classes

segmentor_mapping = {"Cellpose": CustomCellpose}
classifier_mapping = {
    "PatchClassifier": PatchClassifier,
    "RandomForest": FeatureClassifier,
}


class Inst2MultiSeg(Model):
    """A two stage model for: 1. instance segmentation and 2. object wise classification"""

    def __init__(
        self,
        model_name: str,
        model_config: dict,
        data_config: dict,
        eval_config: dict,
    ) -> None:
        """Constructs all the necessary attributes for the Inst2MultiSeg

        :param model_name: Name of the model.
        :type model_name: str
        :param model_config: Model configuration.
        :type model_config: dict
        :param data_config: Data configurations
        :type data_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        """
        # super().__init__()
        Model.__init__(
            self, model_name, model_config, data_config, eval_config
        )

        self.model_name = model_name
        self.model_config = model_config
        self.data_config = data_config
        self.eval_config = eval_config

        self.segmentor_class = self.model_config.get("segmentor_name", "Cellpose")
        self.classifier_class = self.model_config.get(
            "classifier_name", "PatchClassifier"
        )

        # Initialize the cellpose model and the classifier
        segmentor = segmentor_mapping.get(self.segmentor_class)
        self.segmentor = segmentor(
            self.segmentor_class,
            self.model_config,
            self.data_config,
            self.eval_config,
        )
        classifier = classifier_mapping.get(self.classifier_class)
        self.classifier = classifier(
            self.classifier_class,
            self.model_config,
            self.data_config,
            self.eval_config,
        )

        # make sure include mask is set to False if we are using the random forest model
        if self.classifier_class == "RandomForest":
            if (
                "include_mask" not in self.model_config["classifier"].keys()
                or self.model_config["classifier"]["include_mask"] is True
            ):
                # print("Include mask=True was found, but for Random Forest, this parameter must be set to False. Doing this now.")
                self.model_config["classifier"]["include_mask"] = False

    def eval(self, img: np.ndarray) -> np.ndarray:
        """Evaluate the model on the provided image and return the final mask.

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
                include_mask=self.model_config["classifier"]["include_mask"],
            )
            if self.classifier_class == "RandomForest":
                x = create_dataset_for_rf(x, patch_masks)
            # loop over patches and create classification mask
            for idx in range(len(x)):
                patch_class = self.classifier.eval(x[idx])
                # Assign predicted class to corresponding location in final_mask
                patch_class = (
                    patch_class.item()
                    if isinstance(patch_class, torch.Tensor)
                    else patch_class
                )
                class_mask[instance_mask == instance_labels[idx]] = patch_class + 1
            # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
            final_mask = np.stack(
                (instance_mask, class_mask), axis=self.eval_config["mask_channel_axis"]
            ).astype(
                np.uint16
            )  # size 2xHxW

        return final_mask
