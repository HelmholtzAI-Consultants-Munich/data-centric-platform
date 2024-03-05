from typing import List
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

#from dcp_server.models import Model
from dcp_server.utils.processing import normalise

class UNet(nn.Module): # Model

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

        def __init__(self,
                     in_channels: int, 
                     out_channels: int
                     ) -> None:
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

        def forward(self,
                    x: torch.Tensor
                    ) -> torch.Tensor:
            """Forward pass through the DoubleConv module.

            :param x: Input tensor.
            :type x: torch.Tensor
            """
            return self.conv(x)
    

    def __init__(self,
                 model_config: dict,
                 train_config: dict,
                 eval_config: dict,
                 model_name: str
                 ) -> None:
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

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
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

    def train(self,
              imgs: List[np.ndarray],
              masks: List[np.ndarray]
              ) -> None:
        """
        Trains the UNet model using the provided images and masks.

        :param imgs: Input images for training.
        :type imgs: list[numpy.ndarray]
        :param masks: Masks corresponding to the input images.
        :type masks: list[numpy.ndarray]
        """

        lr = self.train_config["classifier"]["lr"]
        epochs = self.train_config["classifier"]["n_epochs"]
        batch_size = self.train_config["classifier"]["batch_size"]

        # Convert input images and labels to tensors
        # normalize images 
        imgs = [normalise(img) for img in imgs]
        # convert to tensor
        imgs = torch.stack([
            torch.from_numpy(img.astype(np.float32)) for img in imgs
        ])
        imgs = imgs.unsqueeze(1) if imgs.ndim == 3 else imgs
      
        # Classification label mask
        masks = np.array(masks)
        masks = torch.stack([
            torch.from_numpy(mask[1].astype(np.int16)) for mask in masks
        ])
        
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

    def eval(self,
             img: np.ndarray
             ) -> np.ndarray:
        """
        Evaluate the model on the provided image and return the predicted label.
          
        :param img: Input image for evaluation.
        :type img:  np.ndarray[np.uint8]
        :return: predicted mask consists of instance and class masks
        :rtype: numpy.ndarray
        """ 
        with torch.no_grad():
            # normalise
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = torch.from_numpy(img).float().unsqueeze(0)

            img = img.unsqueeze(1) if img.ndim == 3 else img
        
            preds = self.forward(img)
            class_mask = torch.argmax(preds, 1).numpy()[0]

            instance_mask = label((class_mask > 0).astype(int))[0]

            final_mask = np.stack(
                (instance_mask, class_mask), 
                axis=self.eval_config['mask_channel_axis']
            ).astype(
                np.uint16
            ) 

        return final_mask
