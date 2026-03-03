from __future__ import annotations
import bentoml
from typing import List
import numpy as np

from dcp_server.models import CustomCellpose

class CustomRunnable:
    """
    BentoML, Runner represents a unit of computation that can be executed on a remote Python worker and scales independently.
    CustomRunnable is a custom runner defined to meet all the requirements needed for this project.
    """

    SUPPORTED_RESOURCES = ("cpu", "nvidia.com/gpu")  # Support both CPU and GPU
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, name:str, model: CustomCellpose) -> None:
        """Constructs all the necessary attributes for the CustomRunnable.

        :param model: model to be trained or evaluated - will be one of classes in models.py
        """

        self.name = name
        self.model = model

    async def evaluate(self, img: np.ndarray) -> np.ndarray:
        """Evaluate the model - find mask of the given image

        :param img: image to evaluate on
        :type img: np.ndarray
        :param z_axis: z dimension (optional, default is None)
        :type z_axis: int
        :return: mask of the image, list of 2D arrays, or single 3D array (if do_3D=True) labelled image.
        :rtype: np.ndarray
        """
        mask = self.model.eval(img=img)

        return mask
