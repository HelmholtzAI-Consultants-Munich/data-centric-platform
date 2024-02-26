import numpy as np
from skimage.measure import label as label_mask

from dcp_server.models import Model, CustomCellposeModel

class MultiCellpose(Model):
    '''
    Multichannel image segmentation model.
    Run the separate CustomCellposeModel models for each channel return the mask corresponding to each object type.
    '''

    def __init__(self, model_config, train_config, eval_config, model_name="Cellpose"):
        """Constructs all the necessary attributes for the MultiCellpose model.
   
        :param model_config: Model configuration.
        :type model_config: dict
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        :param model_name: Name of the model.
        :type model_name: str
        """

        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.model_name = model_name
        self.num_of_channels = self.model_config["classifier"]["num_classes"]

        self.cellpose_models = [
            CustomCellposeModel(self.model_config, 
                                self.train_config,
                                self.eval_config,
                                self.model_name
            ) for _ in range(self.num_of_channels)
        ]  

    def train(self, imgs, masks):
        """
        Train the model on the provided images and masks.

        :param imgs: Input images for training.
        :type imgs: list[numpy.ndarray]
        :param masks: Masks corresponding to the input images.
        :type masks: list[numpy.ndarray]
        """

        for i in range(self.num_of_channels):

            masks_class = []

            for mask in masks:
                mask_class = mask.copy()
                # set all instances in the instance mask not corresponding to the class in question to zero
                mask_class[0][mask_class[1]!=(i+1)] = 0
                masks_class.append(mask_class)
            
            self.cellpose_models[i].train(imgs, masks_class)

        self.metric = np.mean([self.cellpose_models[i].metric for i in range(self.num_of_channels)])
        self.loss = np.mean([self.cellpose_models[i].loss for i in range(self.num_of_channels)])


    def eval(self, img):
        """Evaluate the model on the provided image. The instance mask are computed as the union of the predicted model outputs, while the class of
        each object is assigned based on majority voting between the models.

        :param img: Input image for evaluation.
        :type img:  np.ndarray[np.uint8]
        :return: predicted mask consists of instance and class masks
        :rtype: numpy.ndarray
        """

        instance_masks, class_masks, model_confidences = [], [], []

        for i in range(self.num_of_channels):
            # get the instance mask and pixel-wise cell probability mask
            instance_mask, probs, _  = self.cellpose_models[i].eval_all_outputs(img)
            confidence = probs[2]
            # assign the appropriate class to all objects detected by this model
            class_mask = np.zeros_like(instance_mask)
            class_mask[instance_mask>0]=(i + 1)
                        
            instance_masks.append(instance_mask)
            class_masks.append(class_mask)
            model_confidences.append(confidence)
        # merge the outputs of the different models using the pixel-wise cell probability mask
        merged_mask_instances, class_mask = self.merge_masks(instance_masks, class_masks, model_confidences)
        # set all connected components to the same label in the instance mask
        instance_mask = label_mask(merged_mask_instances>0)
        # and set the class with the most pixels to that object
        for inst_id in np.unique(instance_mask)[1:]:   
            where_inst_id = np.where(instance_mask==inst_id)
            vals, counts = np.unique(class_mask[where_inst_id], return_counts=True)
            class_mask[where_inst_id] = vals[np.argmax(counts)]
        # take the final mask by stancking instance and class mask
        final_mask = np.stack((instance_mask, class_mask), axis=self.eval_config['mask_channel_axis']).astype(np.uint16)
        
        return final_mask
    
    def merge_masks(self, inst_masks, class_masks, probabilities):
        """Merges the instance and class masks resulting from the different models using the pixel-wise cell probability. The output of the model
        with the maximum probability is selected for each pixel.

        :param inst_masks: List of predicted instance masks from each model.
        :type inst_masks:  List[np.array]
        :param class_masks: List of corresponding class masks from each model.
        :type class_masks:  List[np.array]
        :param probabilities: List of corresponding pixel-wise cell probability masks
        :type probabilities:  List[np.array]
        :return: A tuple containing the following elements:
            - final_mask_inst (numpy.ndarray): A single instance mask where for each pixel the output of the model with the highest probability is selected
            - final_mask_class (numpy.ndarray): A single class mask where for each pixel the output of the model with the highest probability is selected
        :rtype: tuple
        """
        # Convert lists to numpy arrays
        inst_masks = np.array(inst_masks)
        class_masks = np.array(class_masks)
        probabilities = np.array(probabilities)
        
        # Find the index of the mask with the maximum probability for each pixel
        max_prob_indices = np.argmax(probabilities, axis=0)
        
        # Use the index to select the corresponding mask for each pixel
        final_mask_inst = inst_masks[max_prob_indices, np.arange(inst_masks.shape[1])[:, None], np.arange(inst_masks.shape[2])]
        final_mask_class = class_masks[max_prob_indices, np.arange(class_masks.shape[1])[:, None], np.arange(class_masks.shape[2])]

        return final_mask_inst, final_mask_class

