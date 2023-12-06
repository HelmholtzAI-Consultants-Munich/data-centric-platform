from cellpose import models, utils
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np

from cellpose.metrics import aggregated_jaccard_index

#from segment_anything import SamPredictor, sam_model_registry
#from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from dcp_server.utils import get_centered_patches, find_max_patch_size, create_patch_dataset

class CustomCellposeModel(models.CellposeModel, nn.Module):
    """Custom cellpose model inheriting the attributes and functions from the original CellposeModel and implementing
    additional attributes and methods needed for this project.
    """    
    def __init__(self, model_config, train_config, eval_config):
        """Constructs all the necessary attributes for the CustomCellposeModel. 
        The model inherits all attributes from the parent class, the init allows to pass any other argument that the parent class accepts.
        Please, visit here https://cellpose.readthedocs.io/en/latest/api.html#id4 for more details on arguments accepted. 

        :param model_config: dictionary passed from the config file with all the arguments for the __init__ function and model initialization
        :type model_config: dict
        :param train_config: dictionary passed from the config file with all the arguments for training function
        :type train_config: dict
        :param eval_config: dictionary passed from the config file with all the arguments for eval function
        :type eval_config: dict
        """
        
        # Initialize the cellpose model
        #super().__init__(**model_config["segmentor"])
        nn.Module.__init__(self)
        models.CellposeModel.__init__(self, **model_config["segmentor"])
        self.mkldnn = False # otherwise we get error with saving model
        self.train_config = train_config
        self.eval_config = eval_config

    def update_configs(self, train_config, eval_config):
        self.train_config = train_config
        self.eval_config = eval_config
        
    def eval(self, img):
        """Evaluate the model - find mask of the given image
        Calls the original eval function. 

        :param img: image to evaluate on
        :type img: np.ndarray
        :return: mask of the image, list of 2D arrays, or single 3D array (if do_3D=True) labelled image.
        :rtype: np.ndarray
        """  
        return super().eval(x=img, **self.eval_config["segmentor"])[0] # 0 to take only mask

    def train(self, imgs, masks):
        """Trains the given model
        Calls the original train function.

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray]
        """  

        if not isinstance(masks, np.ndarray):
            masks = np.array(masks) 
            
        if masks[0].shape[0] == 2:
            masks = list(masks[:,0,...]) 

        super().train(train_data=deepcopy(imgs), train_labels=masks, **self.train_config["segmentor"])
        
        pred_masks = [self.eval(img) for img in masks]
        self.metric = np.mean(aggregated_jaccard_index(masks, pred_masks))
        # self.loss = self.loss_fn(masks, pred_masks)
    
    def masks_to_outlines(self, mask):
        """ get outlines of masks as a 0-1 array
        Calls the original cellpose.utils.masks_to_outlines function

        :param mask: int, 2D or 3D array, mask of an image
        :type mask: ndarray
        :return: outlines
        :rtype: ndarray
        """        
        return utils.masks_to_outlines(mask) #[True, False] outputs


class CellClassifierFCNN(nn.Module):
    
    '''
    Fully convolutional classifier for cell images. NOTE -> This model cannot be used as a standalone model in DCP

    Args:
        model_config (dict): Model configuration.
        train_config (dict): Training configuration.
        eval_config (dict): Evaluation configuration.
        
    '''

    def __init__(self, model_config, train_config, eval_config):
        super().__init__()

        self.in_channels = model_config["classifier"]["in_channels"]
        self.num_classes = model_config["classifier"]["num_classes"]

        self.train_config = train_config["classifier"]
        self.eval_config = eval_config["classifier"]
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, 2, 5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
        )
        self.final_conv = nn.Conv2d(128, self.num_classes, 1)
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def update_configs(self, train_config, eval_config):
        self.train_config = train_config
        self.eval_config = eval_config

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_conv(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return x

    def train (self, imgs, labels):        
        """
            input:
            1) imgs - List[np.ndarray[np.uint8]] with shape (3, dx, dy)
            2) labels - List[int]
        """

        lr = self.train_config['lr']
        epochs = self.train_config['n_epochs']
        batch_size = self.train_config['batch_size']
        # optimizer_class = self.train_config['optimizer']

        # Convert input images and labels to tensors

        # normalize images 
        imgs = [(img-np.min(img))/(np.max(img)-np.min(img)) for img in imgs]
        # convert to tensor
        imgs = torch.stack([torch.from_numpy(img.astype(np.float32)) for img in imgs])
        imgs = torch.permute(imgs, (0, 3, 1, 2)) 
        # Your classification label mask
        labels = torch.LongTensor([label for label in labels])

        # Create a training dataset and dataloader
        train_dataset = TensorDataset(imgs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(params=self.parameters(), lr=lr) #eval(f'{optimizer_class}(params={self.parameters()}, lr={lr})')
        # TODO check if we should replace self.parameters with super.parameters()
        
        for _ in tqdm(range(epochs), desc="Running CellClassifierFCNN training"):
            self.loss = 0
            for data in train_dataloader:
                imgs, labels = data

                optimizer.zero_grad()
                preds = self.forward(imgs)
            
                l = loss_fn(preds, labels)
                l.backward()
                optimizer.step()
                self.loss += l.item()

            self.loss /= len(train_dataloader) 
    
    def eval(self, img):
        """
        Evaluate the model on the provided image and return the predicted label.
            Input:
            img: np.ndarray[np.uint8]
            Output: y_hat - The predicted label
        """ 
        # normalise
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        # convert to tensor
        img = torch.permute(torch.tensor(img.astype(np.float32)), (2, 0, 1)).unsqueeze(0) 
        preds = self.forward(img)
        y_hat = torch.argmax(preds, 1)
        return y_hat


class CellposePatchCNN(nn.Module):

    """
    Cellpose & patches of cells and then cnn to classify each patch
    """
    
    def __init__(self, model_config, train_config, eval_config):
        super().__init__()

        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config

        # Initialize the cellpose model and the classifier
        self.segmentor = CustomCellposeModel(self.model_config, 
                                             self.train_config,
                                             self.eval_config)
        self.classifier = CellClassifierFCNN(self.model_config,
                                             self.train_config,
                                             self.eval_config)

    def update_configs(self, train_config, eval_config):
        self.train_config = train_config
        self.eval_config = eval_config
        
    def train(self, imgs, masks):
        """Trains the given model. First trains the segmentor and then the clasiffier.

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray] of same shape as output of eval, i.e. one channel instances, 
        second channel classes, so [2, H, W] or [2, 3, H, W] for 3D
        """  
        # train cellpose
        masks = np.array(masks) 
        masks_instances = list(masks[:,0,...]) #[mask.sum(-1) for mask in masks] if masks[0].ndim == 3 else masks
        self.segmentor.train(imgs, masks_instances)
        # create patch dataset to train classifier
        masks_classes = list(masks[:,1,...]) #[((mask > 0) * np.arange(1, 4)).sum(-1) for mask in masks]
        patches, labels = create_patch_dataset(imgs,
                                               masks_classes,
                                               masks_instances,
                                               noise_intensity = self.train_config["classifier"]["train_data"]["noise_intensity"],
                                               max_patch_size = self.train_config["classifier"]["train_data"]["patch_size"])
        # train classifier
        self.classifier.train(patches, labels)

    def eval(self, img):
        # TBD we assume image is either 2D [H, W] (see fsimage storage)
        # The final mask which is returned should have 
        # first channel the output of cellpose and the rest are the class channels
        with torch.no_grad():
            # get instance mask from segmentor
            instance_mask = self.segmentor.eval(img)
            # find coordinates of detected objects
            class_mask = np.zeros(instance_mask.shape)
            
            max_patch_size = self.eval_config["classifier"]["data"]["patch_size"]
            if max_patch_size is None: max_patch_size = find_max_patch_size(instance_mask)
            noise_intensity = self.eval_config["classifier"]["data"]["noise_intensity"]
            
            # get patches centered around detected objects
            patches, instance_labels, _ = get_centered_patches(img,
                                                               instance_mask,
                                                               max_patch_size,
                                                               noise_intensity=noise_intensity)
            # loop over patches and create classification mask
            for idx, patch in enumerate(patches):
                patch_class = self.classifier.eval(patch) # patch size should be HxWxC, e.g. 64,64,3
                # Assign predicted class to corresponding location in final_mask
                class_mask[instance_mask==instance_labels[idx]] = patch_class.item() + 1
            # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
            #class_mask = class_mask * (instance_mask > 0)#.long())
            final_mask = np.stack((instance_mask, class_mask), axis=self.eval_config['mask_channel_axis']).astype(np.uint16) # size 2xHxW
        
        return final_mask


        
# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
