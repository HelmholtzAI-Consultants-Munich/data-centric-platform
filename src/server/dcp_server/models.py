from cellpose import models, utils
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import F1Score
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.exceptions import NotFittedError

from cellpose.metrics import aggregated_jaccard_index
from cellpose.dynamics import labels_to_flows
#from segment_anything import SamPredictor, sam_model_registry
#from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator

from dcp_server.utils import get_centered_patches, find_max_patch_size, create_patch_dataset, create_dataset_for_rf

class CustomCellposeModel(models.CellposeModel, nn.Module):
    """Custom cellpose model inheriting the attributes and functions from the original CellposeModel and implementing
    additional attributes and methods needed for this project.
    """    
    def __init__(self, model_config, train_config, eval_config, model_name):
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
        self.loss = 1e6
        self.model_name = model_name

    def update_configs(self, train_config, eval_config):
        self.train_config = train_config
        self.eval_config = eval_config

    def eval_probas(self, img):

        return super().eval(x=img, **self.eval_config["segmentor"])[1][2]

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

        if not isinstance(masks, np.ndarray): # TODO Remove: all these should be taken care of in fsimagestorage
            masks = np.array(masks) 
            
        if masks[0].shape[0] == 2:
            masks = list(masks[:,0,...]) 
        super().train(train_data=deepcopy(imgs), train_labels=masks, **self.train_config["segmentor"])

        # compute loss and metric
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
        # compute loss, combination of mse for flows and bce for cell probability
        self.loss = self.loss_fn(true_lbl, pred_flows) 
        self.metric = np.mean(aggregated_jaccard_index(masks, pred_masks))
    
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

        self.in_channels = model_config["classifier"].get("in_channels",1)
        self.num_classes = model_config["classifier"].get("num_classes",3)

        self.train_config = train_config["classifier"]
        self.eval_config = eval_config["classifier"]

        self.include_mask = model_config["classifier"]["include_mask"]
        self.in_channels = self.in_channels + 1 if self.include_mask else self.in_channels
        
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

        self.metric_fn = F1Score(num_classes=self.num_classes, task="multiclass") 

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
            self.loss, self.metric = 0, 0
            for data in train_dataloader:
                imgs, labels = data

                optimizer.zero_grad()
                preds = self.forward(imgs)
            
                l = loss_fn(preds, labels)
                l.backward()
                optimizer.step()
                self.loss += l.item()

                self.metric += self.metric_fn(preds, labels)

            self.loss /= len(train_dataloader) 
            self.metric /= len(train_dataloader)
    
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
    
    def __init__(self, model_config, train_config, eval_config, model_name):
        super().__init__()

        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.include_mask = self.model_config["classifier"]["include_mask"]
        self.model_name = model_name
        self.classifier_class = self.model_config.get("classifier").get("model_class", "CellClassifierFCNN")

        # Initialize the cellpose model and the classifier
        self.segmentor = CustomCellposeModel(self.model_config, 
                                             self.train_config,
                                             self.eval_config,
                                             "Cellpose")
        
        if self.classifier_class == "FCNN":
            self.classifier = CellClassifierFCNN(self.model_config,
                                                 self.train_config,
                                                 self.eval_config)
            
        elif self.classifier_class == "RandomForest":
            self.classifier = CellClassifierShallowModel(self.model_config,
                                                         self.train_config,
                                                         self.eval_config)
            # make sure include mask is set to False if we are using the random forest model 
            self.include_mask = False 
            
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
        self.segmentor.train(deepcopy(imgs), masks_instances)
        # create patch dataset to train classifier
        masks_classes = list(masks[:,1,...]) #[((mask > 0) * np.arange(1, 4)).sum(-1) for mask in masks]
        patches, patch_masks, labels = create_patch_dataset(imgs,
                                                            masks_classes,
                                                            masks_instances,
                                                            noise_intensity = self.train_config["classifier"]["train_data"]["noise_intensity"],
                                                            max_patch_size = self.train_config["classifier"]["train_data"]["patch_size"],
                                                            include_mask = self.include_mask)
        x = patches
        if self.classifier_class == "RandomForest":
            x = create_dataset_for_rf(patches, patch_masks)
        # train classifier
        self.classifier.train(x, labels)
        # and compute metric and loss
        self.metric = (self.segmentor.metric + self.classifier.metric) / 2
        self.loss = (self.segmentor.loss + self.classifier.loss)/2

    def eval(self, img):
        # TBD we assume image is 2D [H, W] (see fsimage storage)
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
            patches, patch_masks, instance_labels, _ = get_centered_patches(img,
                                                               instance_mask,
                                                               max_patch_size,
                                                               noise_intensity=noise_intensity,
                                                               include_mask=self.include_mask)
            x = patches
            if self.classifier_class == "RandomForest":
                x = create_dataset_for_rf(patches, patch_masks)
            # loop over patches and create classification mask
            for idx in range(len(x)):
                patch_class = self.classifier.eval(x[idx])
                # Assign predicted class to corresponding location in final_mask
                patch_class = patch_class.item() if isinstance(patch_class, torch.Tensor) else patch_class
                class_mask[instance_mask==instance_labels[idx]] = patch_class + 1
            # Apply mask to final_mask, retaining only regions where cellpose_mask is greater than 0
            final_mask = np.stack((instance_mask, class_mask), axis=self.eval_config['mask_channel_axis']).astype(np.uint16) # size 2xHxW
        
        return final_mask

class CellClassifierShallowModel:

    def __init__(self, model_config, train_config, eval_config):

        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config

        self.model = RandomForestClassifier() # TODO chnage config so RandomForestClassifier accepts input params

   
    def train(self, X_train, y_train):

        self.model.fit(X_train,y_train)

        y_hat = self.model.predict(X_train)
        y_hat_proba = self.model.predict_proba(X_train)

        self.metric = f1_score(y_train, y_hat, average='micro')
        # Binary Cross Entrop Loss
        self.loss = log_loss(y_train, y_hat_proba)

    
    def eval(self, X_test):

        X_test = X_test.reshape(1,-1)

        try:
            y_hat = self.model.predict(X_test)
        except NotFittedError as e:
            y_hat = np.zeros(X_test.shape[0])
   
        return y_hat

class UNet(nn.Module):

    """
    Unet is a convolutional neural network architecture for semantic segmentation.
    
    Args:
        in_channels (int): Number of input channels (default: 3).
        out_channels (int): Number of output channels (default: 4).
        features (list): List of feature channels for each encoder level (default: [64,128,256,512]).
    """
    
    class DoubleConv(nn.Module):
        """
        DoubleConv module consists of two consecutive convolutional layers with
        batch normalization and ReLU activation functions.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        def __init__(self, in_channels, out_channels):
            super().__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.conv(x)
    

    def __init__(self, model_config, train_config, eval_config, model_name):

        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config
        self.model_name = model_name
        '''
        self.in_channels = self.model_config["unet"]["in_channels"]
        self.out_channels = self.model_config["unet"]["out_channels"]
        self.features = self.model_config["unet"]["features"]
        '''
        self.in_channels = self.model_config["classifier"]["in_channels"]
        self.out_channels = self.model_config["classifier"]["num_classes"] + 1
        self.features = self.model_config["classifier"]["features"]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in self.features:
            self.encoder.append(
                UNet.DoubleConv(self.in_channels, feature)
            )
            self.in_channels = feature

        # Decoder
        for feature in self.features[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.decoder.append(
                UNet.DoubleConv(feature*2, feature)
            )

        self.bottle_neck = UNet.DoubleConv(self.features[-1], self.features[-1]*2)
        self.output_conv = nn.Conv2d(self.features[0], self.out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottle_neck(x)
        skip_connections = skip_connections[::-1]

        for i in np.arange(len(self.decoder), step=2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[i//2]
            concatenate_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i+1](concatenate_skip)

        return self.output_conv(x)

    def train(self, imgs, masks):

        lr = self.train_config["classifier"]['lr']
        epochs = self.train_config["classifier"]['n_epochs']
        batch_size = self.train_config["classifier"]['batch_size']

        # Convert input images and labels to tensors
        # normalize images 
        imgs = [(img-np.min(img))/(np.max(img)-np.min(img)) for img in imgs]
        # convert to tensor
        imgs = torch.stack([torch.from_numpy(img.astype(np.float32)) for img in imgs])
        imgs = imgs.unsqueeze(1) if imgs.ndim == 3 else imgs
      
        # Classification label mask
        masks = np.array(masks)
        masks = torch.stack([torch.from_numpy(mask[1]) for mask in masks])

        # Create a training dataset and dataloader
        train_dataset = TensorDataset(imgs, masks)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(params=self.parameters(), lr=lr)

        for _ in tqdm(range(epochs), desc="Running UNet training"):

            self.loss = 0

            for imgs, masks in train_dataloader:
                imgs = imgs.float()
                masks = masks.long()

                #forward path
                preds = self.forward(imgs)
                loss = loss_fn(preds, masks)

                #backward path
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.loss += loss.detach().mean().item()

            self.loss /= len(train_dataloader) 

    def eval(self, img):
        """
        Evaluate the model on the provided image and return the predicted label.
            Input:
            img: np.ndarray[np.uint8]
            Output: y_hat - The predicted label
        """ 
        with torch.no_grad():
            # normalise
            img = (img-np.min(img))/(np.max(img)-np.min(img))
            img = torch.from_numpy(img).float().unsqueeze(0)

            img = img.unsqueeze(1) if img.ndim == 3 else img
        
            preds = self.forward(img)
            class_mask = torch.argmax(preds, 1).numpy()[0]

            instance_mask = label((class_mask > 0).astype(int))[0]

            final_mask = np.stack((instance_mask, class_mask), axis=self.eval_config['mask_channel_axis']).astype(np.uint16) 

        return final_mask

class CellposeMultichannel():
    '''
    Multichannel image segmentation model.
    Run the separate cellpose model for each channel return the mask corresponding to each object type.

    Args:
        num_of_channels (int): Number of channels in the input image.
        device: The device on which the models should run (e.g., 'cuda' for GPU or 'cpu' for CPU).

    '''

    def __init__(self, model_config, train_config, eval_config, model_name="Cellpose"):

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

        for i in range(self.num_of_channels):

            masks_class = []

            for mask in masks:
                mask_class = mask.copy()
                mask_class[0][mask_class[1]!=(i+1)] = 0
                masks_class.append(mask_class)

            self.cellpose_models[i].train(imgs, masks_class)

        self.metric = np.mean([self.cellpose_models[i].metric for i in range(self.num_of_channels)])
        self.loss = np.mean([self.cellpose_models[i].loss for i in range(self.num_of_channels)])


    def eval(self, img):

        instance_masks, class_masks, class_probas = [], [], []

        instance_offset = 0

        for i in range(self.num_of_channels):

            res = self.cellpose_models[i].eval(img)
            res[res>0] += instance_offset
            instance_masks.append(res)

            instance_offset = np.max(res)

            label_mask = res.copy()
            label_mask[res>0]=(i + 1)
            class_masks.append(label_mask)

            class_proba = self.cellpose_models[i].eval_probas(img)
            class_probas.append(class_proba)

        instance_mask = sum(instance_masks)
        class_probas = np.argmax(np.stack(class_probas), axis=0)

        # merge the outputs of the n models
        class_masks = np.stack(class_masks)
        indexes = class_probas*class_probas.size + np.arange(class_probas.size).reshape(class_probas.shape)
        indexes = np.unravel_index(indexes, class_masks.shape)
        class_mask = class_masks[indexes]

        final_mask = np.stack((instance_mask, class_mask), axis=self.eval_config['mask_channel_axis']).astype(np.uint16)
        
        return final_mask


    
    #    return np.sum([self.cellpose_models[i].eval(img) for i in range(self.num_of_channels)])



# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
