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


class PatchClassifier(nn.Module):
    
    """ Fully convolutional classifier for cell images. NOTE -> This model cannot be used as a standalone model in DCP        
    """

    def __init__(self,
                 model_name: str,
                 model_config: dict,
                 data_config: dict,
                 train_config: dict,
                 eval_config: dict
                 ) -> None:
        """ Initialize the fully convolutional classifier.

        :param model_name: Name of the model.
        :type model_name: str
        :param model_config: Model configuration.
        :type model_config: dict
        :param data_config: Data configuration.
        :type data_config: dict       
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        """
        super().__init__()


        self.model_name = model_name
        self.model_config = model_config["classifier"]
        self.data_config = data_config
        self.train_config = train_config["classifier"]
        self.eval_config = eval_config["classifier"]
        
        self.build_model()

    def train (self,
               imgs: List[np.ndarray],
               labels: List[np.ndarray]
               ) -> None:        
        """ Trains the given model

        :param imgs: List of input images with shape (3, dx, dy).
        :type imgs: List[np.ndarray[np.uint8]]
        :param labels: List of classification labels.
        :type labels: List[int]
        """

        # Convert input images and labels to tensors
        imgs = torch.stack([torch.from_numpy(img.astype(np.float32)) for img in imgs])
        imgs = torch.permute(imgs, (0, 3, 1, 2)) 
        # Your classification label mask
        labels = torch.LongTensor([label for label in labels])

        # Create a training dataset and dataloader
        train_dataloader = DataLoader(
            TensorDataset(imgs, labels),
            batch_size=self.train_config["batch_size"])

        loss_fn = nn.CrossEntropyLoss()
        optimizer = Adam(
            params=self.parameters(),
            lr=self.train_config["lr"]
            ) 
        # optimizer_class = self.train_config["optimizer"]
        #eval(f'{optimizer_class}(params={self.parameters()}, lr={lr})')
        
        # TODO check if we should replace self.parameters with super.parameters()
        
        for _ in tqdm(
            range(self.train_config["n_epochs"]),
            desc="Running PatchClassifier training"
            ):
            
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
        """ Evaluates the model on the provided image and return the predicted label.

        :param img: Input image for evaluation.
        :type img: np.ndarray[np.uint8]
        :return: y_hat - predicted label.
        :rtype: torch.Tensor
        """
        # convert to tensor
        img = torch.permute(torch.tensor(img.astype(np.float32)), (2, 0, 1)).unsqueeze(0) 
        preds = self.forward(img)
        y_hat = torch.argmax(preds, 1)
        return y_hat

    def build_model(self):
        """ Builds the PatchClassifer.
        """
        in_channels = self.model_config["in_channels"]
        in_channels = in_channels + 1 if self.model_config["include_mask"] else in_channels
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 5),
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
        self.final_conv = nn.Conv2d(128,
                                    self.model_config["num_classes"],
                                    1)
        self.pooling = nn.AdaptiveMaxPool2d(1)

        self.metric_fn = F1Score(num_classes=self.model_config["num_classes"],
                                 task="multiclass") 

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """ Performs forward pass of the PatchClassifier.

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


class FeatureClassifier:
    """ This class implements a shallow model for cell classification using scikit-learn.
    """

    def __init__(self,
                 model_name: str,
                 model_config: dict,
                 data_config: dict,
                 train_config: dict,
                 eval_config: dict
                 ) -> None:
        """ Constructs all the necessary attributes for the FeatureClassifier

        :param model_config: Model configuration.
        :type model_config: dict
        :param data_config: Data configuration.
        :type data_config: dict
        :param train_config: Training configuration.
        :type train_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        """

        self.model_name = model_name
        self.model_config = model_config # use for initialising model
        self.data_config = data_config
        self.train_config = train_config
        self.eval_config = eval_config

        self.model = RandomForestClassifier() # TODO chnage config so RandomForestClassifier accepts input params

   
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray
              ) -> None:
        """ Trains the model using the provided training data.

        :param X_train: Features of the training data.
        :type X_train: numpy.ndarray
        :param y_train: Labels of the training data.
        :type y_train: numpy.ndarray
        """

        self.model.fit(X_train,y_train)

        y_hat = self.model.predict(X_train)
        y_hat_proba = self.model.predict_proba(X_train)

        # Binary Cross Entrop Loss
        self.loss = log_loss(y_train, y_hat_proba)
        self.metric = f1_score(y_train, y_hat, average='micro')

    
    def eval(self,
             X_test: np.ndarray
             ) -> np.ndarray:
        """ Evaluates the model on the provided test data.

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
