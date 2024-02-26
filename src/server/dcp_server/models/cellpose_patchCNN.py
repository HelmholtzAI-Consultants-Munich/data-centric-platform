from copy import deepcopy
from tqdm import tqdm
from typing import List
import numpy as np

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import F1Score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.exceptions import NotFittedError

from dcp_server.models import Model, CustomCellposeModel
from dcp_server.utils.processing import (
    normalise,
    get_centered_patches,
    find_max_patch_size,
    create_patch_dataset,
    create_dataset_for_rf
)


class CellposePatchCNN(Model):
    """
    Cellpose & patches of cells and then cnn to classify each patch
    """
    
    def __init__(self,
                 model_config: dict,
                 train_config: dict,
                 eval_config:dict,
                 model_name:str
                 ) -> None:
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
        self.model_name = model_name
        self.include_mask = self.model_config["classifier"]["include_mask"]
        self.classifier_class = self.model_config.get("classifier").get("model_class", "CellClassifierFCNN")

        # Initialize the cellpose model and the classifier
        self.segmentor = CustomCellposeModel(
            self.model_config, self.train_config, self.eval_config, "Cellpose"
            )
        
        if self.classifier_class == "FCNN":
            self.classifier = CellClassifierFCNN(
                self.model_config, self.train_config, self.eval_config
                )
            
        elif self.classifier_class == "RandomForest":
            self.classifier = CellClassifierShallowModel(
                self.model_config, self.train_config, self.eval_config
                )
            # make sure include mask is set to False if we are using the random forest model 
            self.include_mask = False 
            
        
    def train(self,
              imgs: List[np.ndarray],
              masks: List[np.ndarray]
              ) -> None:
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
        masks_classes = list(
            masks[:,1,...]
        ) #[((mask > 0) * np.arange(1, 4)).sum(-1) for mask in masks]
        patches, patch_masks, labels = create_patch_dataset(
            imgs,
            masks_classes,
            masks_instances,
            noise_intensity = self.train_config["classifier"]["train_data"]["noise_intensity"],
            max_patch_size = self.train_config["classifier"]["train_data"]["patch_size"],
            include_mask = self.include_mask
        )
        x = patches
        if self.classifier_class == "RandomForest":
            x = create_dataset_for_rf(patches, patch_masks)
        # train classifier
        self.classifier.train(x, labels)
        # and compute metric and loss
        self.metric = (self.segmentor.metric + self.classifier.metric) / 2
        self.loss = (self.segmentor.loss + self.classifier.loss)/2

    def eval(self,
             img: np.ndarray
             ) -> np.ndarray:
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
            if max_patch_size is None: 
                max_patch_size = find_max_patch_size(instance_mask)
            noise_intensity = self.eval_config["classifier"]["data"]["noise_intensity"]
            
            # get patches centered around detected objects
            patches, patch_masks, instance_labels, _ = get_centered_patches(
                img,
                instance_mask,
                max_patch_size,
                noise_intensity=noise_intensity,
                include_mask=self.include_mask
            )
            x = patches
            if self.classifier_class == "RandomForest":
                x = create_dataset_for_rf(patches, patch_masks)
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


class CellClassifierFCNN(nn.Module):
    
    """Fully convolutional classifier for cell images. NOTE -> This model cannot be used as a standalone model in DCP

    :param model_config: Model configuration.
    :type model_config: dict
    :param train_config: Training configuration.
    :type train_config: dict
    :param eval_config: Evaluation configuration.
    :type eval_config: dict
        
    """

    def __init__(self,
                 model_config: dict,
                 train_config: dict,
                 eval_config: dict
                 ) -> None:
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

    def update_configs(self,
                       train_config: dict,
                       eval_config: dict
                       ) -> None:
        """
        Update the training and evaluation configurations.

        :param train_config: Dictionary containing the training configuration.
        :type train_config: dict
        :param eval_config: Dictionary containing the evaluation configuration.
        :type eval_config: dict
        """
        self.train_config = train_config
        self.eval_config = eval_config

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
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

    def train (self,
               imgs: List[np.ndarray],
               labels: List[np.ndarray]
               ) -> None:        
        """Trains the given model

        :param imgs: List of input images with shape (3, dx, dy).
        :type imgs: List[np.ndarray[np.uint8]]
        :param labels: List of classification labels.
        :type labels: List[int]
        """

        lr = self.train_config["lr"]
        epochs = self.train_config["n_epochs"]
        batch_size = self.train_config["batch_size"]
        # optimizer_class = self.train_config["optimizer"]

        # Convert input images and labels to tensors

        # normalize images 
        imgs = [normalise_image(img) for img in imgs]
        # convert to tensor
        imgs = torch.stack([torch.from_numpy(img.astype(np.float32)) for img in imgs])
        imgs = torch.permute(imgs, (0, 3, 1, 2)) 
        # Your classification label mask
        labels = torch.LongTensor([label for label in labels])

        # Create a training dataset and dataloader
        train_dataset = TensorDataset(imgs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(
            params=self.parameters(),
            lr=lr
            ) #eval(f'{optimizer_class}(params={self.parameters()}, lr={lr})')
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
    
    def eval(self,
             img: np.ndarray
             ) -> torch.Tensor:
        """Evaluates the model on the provided image and return the predicted label.

        :param img: Input image for evaluation.
        :type img: np.ndarray[np.uint8]
        :return: y_hat - predicted label.
        :rtype: torch.Tensor
        """
        # normalise
        img = normalise(img)
        # convert to tensor
        img = torch.permute(torch.tensor(img.astype(np.float32)), (2, 0, 1)).unsqueeze(0) 
        preds = self.forward(img)
        y_hat = torch.argmax(preds, 1)
        return y_hat


class CellClassifierShallowModel:
    """
    This class implements a shallow model for cell classification using scikit-learn.
    """

    def __init__(self,
                 model_config: dict,
                 train_config: dict,
                 eval_config: dict
                 ) -> None:
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

   
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray
              ) -> None:
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

    
    def eval(self,
             X_test: np.ndarray
             ) -> np.ndarray:
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
