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
from skimage.measure import label as label_mask


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
        """Update the training and evaluation configurations.

        :param train_config: Dictionary containing the training configuration.
        :type train_config: dict
        :param eval_config: Dictionary containing the evaluation configuration.
        :type eval_config: dict
        """
        self.train_config = train_config
        self.eval_config = eval_config

    def eval_all_outputs(self, img):
        """Get all outputs of the model when running eval.

        :param img: Input image for segmentation.
        :type img: numpy.ndarray
        :return: Probability mask for the input image.
        :rtype: numpy.ndarray
        """

        return super().eval(x=img, **self.eval_config["segmentor"])

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
    
    """Fully convolutional classifier for cell images. NOTE -> This model cannot be used as a standalone model in DCP

    :param model_config: Model configuration.
    :type model_config: dict
    :param train_config: Training configuration.
    :type train_config: dict
    :param eval_config: Evaluation configuration.
    :type eval_config: dict
        
    """

    def __init__(self, model_config, train_config, eval_config):
        """Initialize the fully convolutional classifier.

        :param model_config: Model configuration.
        :type model_config: dict
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        """
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
        """
        Update the training and evaluation configurations.

        :param train_config: Dictionary containing the training configuration.
        :type train_config: dict
        :param eval_config: Dictionary containing the evaluation configuration.
        :type eval_config: dict
        """
        self.train_config = train_config
        self.eval_config = eval_config

    def forward(self, x):
        """ Performs forward pass of the CellClassifierFCNN.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor after passing through the network.
        :rtype: torch.Tensor
        """

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.final_conv(x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return x

    def train (self, imgs, labels):        
        """Trains the given model

        :param imgs: List of input images with shape (3, dx, dy).
        :type imgs: List[np.ndarray[np.uint8]]
        :param labels: List of classification labels.
        :type labels: List[int]
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
        """Evaluates the model on the provided image and return the predicted label.

        :param img: Input image for evaluation.
        :type img: np.ndarray[np.uint8]
        :return: y_hat - predicted label.
        :rtype: torch.Tensor
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
        """Constructs all the necessary attributes for the CellposePatchCNN

        :param model_config: Model configuration.
        :type model_config: dict
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        :param model_name: Name of the model.
        :type model_name: str
    """
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
        """Update the training and evaluation configurations.

        :param train_config: Dictionary containing the training configuration.
        :type train_config: dict
        :param eval_config: Dictionary containing the evaluation configuration.
        :type eval_config: dict
        """
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
    """
    This class implements a shallow model for cell classification using scikit-learn.
    """

    def __init__(self, model_config, train_config, eval_config):
        """Constructs all the necessary attributes for the CellClassifierShallowModel

        :param model_config: Model configuration.
        :type model_config: dict
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        """

        self.model_config = model_config
        self.train_config = train_config
        self.eval_config = eval_config

        self.model = RandomForestClassifier() # TODO chnage config so RandomForestClassifier accepts input params

   
    def train(self, X_train, y_train):
        """Trains the model using the provided training data.

        :param X_train: Features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: Labels of the training data.
        :type y_train: numpy.ndarray
        """

        self.model.fit(X_train,y_train)

        y_hat = self.model.predict(X_train)
        y_hat_proba = self.model.predict_proba(X_train)

        self.metric = f1_score(y_train, y_hat, average='micro')
        # Binary Cross Entrop Loss
        self.loss = log_loss(y_train, y_hat_proba)

    
    def eval(self, X_test):
        """Evaluates the model on the provided test data.

        :param X_test: Features of the test data.
        :type X_test: numpy.ndarray
        :return: y_hat - predicted labels.
        :rtype: numpy.ndarray
        """

        X_test = X_test.reshape(1,-1)

        try:
            y_hat = self.model.predict(X_test)
        except NotFittedError as e:
            y_hat = np.zeros(X_test.shape[0])
   
        return y_hat

class UNet(nn.Module):

    """
    Unet is a convolutional neural network architecture for semantic segmentation.
    
    :param in_channels: Number of input channels (default: 3).
    :type in_channels: int
    :param out_channels: Number of output channels (default: 4).
    :type out_channels: int
    :param features: List of feature channels for each encoder level (default: [64,128,256,512]).
    :type features: list
    """
    
    class DoubleConv(nn.Module):
        """
        DoubleConv module consists of two consecutive convolutional layers with
        batch normalization and ReLU activation functions.
        """

        def __init__(self, in_channels, out_channels):
            """
            Initialize DoubleConv module.

            :param in_channels: Number of input channels.
            :type in_channels: int
            :param out_channels: Number of output channels.
            :type out_channels: int
            """
            
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
            """Forward pass through the DoubleConv module.

            :param x: Input tensor.
            :type x: torch.Tensor
            """
            return self.conv(x)
    

    def __init__(self, model_config, train_config, eval_config, model_name):
        """Constructs all the necessary attributes for the UNet model.
   
   
        :param model_config: Model configuration.
        :type model_config: dict
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        :param model_name: Name of the model.
        :type model_name: str
        """

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
        """
        Forward pass of the UNet model.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
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
        """
        Trains the UNet model using the provided images and masks.

        :param imgs: Input images for training.
        :type imgs: list[numpy.ndarray]
        :param masks: Masks corresponding to the input images.
        :type masks: list[numpy.ndarray]
        """

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
        masks = torch.stack([torch.from_numpy(mask[1].astype(np.int16)) for mask in masks])
        
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
          
        :param img: Input image for evaluation.
        :type img:  np.ndarray[np.uint8]
        :return: predicted mask consists of instance and class masks
        :rtype: numpy.ndarray
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
    Run the separate CustomCellposeModel models for each channel return the mask corresponding to each object type.
    '''

    def __init__(self, model_config, train_config, eval_config, model_name="Cellpose"):
        """Constructs all the necessary attributes for the CellposeMultichannel model.
   
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


    



# class CustomSAMModel():
# # https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb
#     def __init__(self):
#         pass
