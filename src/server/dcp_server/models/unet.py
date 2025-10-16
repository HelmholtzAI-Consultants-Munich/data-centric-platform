from typing import List
from tqdm import tqdm
import numpy as np
from scipy.ndimage import label

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import JaccardIndex

from .model import Model
from dcp_server.utils.processing import convert_to_tensor


class UNet(nn.Module, Model):
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

        def __init__(self, in_channels: int, out_channels: int) -> None:
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the DoubleConv module.

            :param x: Input tensor.
            :type x: torch.Tensor
            """
            return self.conv(x)

    def __init__(
        self,
        model_name: str,
        model_config: dict,
        data_config: dict,
        eval_config: dict,
    ) -> None:
        """Constructs all the necessary attributes for the UNet model.

        :param model_name: Name of the model.
        :type model_name: str
        :param model_config: Model configuration.
        :type model_config: dict
        :param data_config: Data configurations
        :type data_config: dict
        :param eval_config: Evaluation configuration.
        :type eval_config: dict
        """
        Model.__init__(
            self, model_name, model_config, data_config, eval_config
        )
        nn.Module.__init__(self)
        # super().__init__()

        self.model_name = model_name
        self.model_config = model_config
        self.data_config = data_config
        self.eval_config = eval_config

        self.loss = 1e6
        self.metric = 0
        self.num_classes = self.model_config["classifier"]["num_classes"] + 1
        self.metric_f = JaccardIndex(
            task="multiclass", num_classes=self.num_classes, average="macro", ignore_index=0
        )

        self.build_model()

    def eval(self, img: np.ndarray) -> np.ndarray:
        """Evaluate the model on the provided image and return the predicted label.

        :param img: Input image for evaluation.
        :type img:  np.ndarray[np.uint8]
        :return: predicted mask consists of instance and class masks
        :rtype: numpy.ndarray
        """
        with torch.no_grad():

            # img = torch.from_numpy(img).float().unsqueeze(0)
            # img = img.unsqueeze(1) if img.ndim == 3 else img
            img = convert_to_tensor([img], np.float32)

            preds = self.forward(img)
            class_mask = torch.argmax(preds, 1).numpy()[0]
            if self.eval_config["compute_instance"] is True:
                instance_mask = label((class_mask > 0).astype(int))[0]
                final_mask = np.stack(
                    [instance_mask, class_mask],
                    axis=self.eval_config["mask_channel_axis"],
                ).astype(np.uint16)
            else:
                final_mask = class_mask.astype(np.uint16)

        return final_mask

    def build_model(self) -> None:
        """Builds the UNet."""
        in_channels = self.model_config["classifier"]["in_channels"]
        out_channels = self.num_classes
        features = self.model_config["classifier"]["features"]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.encoder.append(UNet.DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in features[::-1]:
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(UNet.DoubleConv(feature * 2, feature))

        self.bottle_neck = UNet.DoubleConv(features[-1], features[-1] * 2)
        self.output_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            skip_connection = skip_connections[i // 2]
            concatenate_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](concatenate_skip)

        return self.output_conv(x)
