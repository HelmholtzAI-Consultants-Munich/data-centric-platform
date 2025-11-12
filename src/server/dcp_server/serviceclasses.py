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

    SUPPORTED_RESOURCES = ("cpu",)  # TODO add here?
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, name:str, model: CustomCellpose, save_model_path: str) -> None:
        """Constructs all the necessary attributes for the CustomRunnable.

        :param model: model to be trained or evaluated - will be one of classes in models.py
        :param save_model_path: full path of the model object that it will be saved into
        :type save_model_path: str
        """

        self.name = name
        self.model = model
        self.save_model_path = save_model_path
        # update with the latest model if it already exists to continue training from there?
        self.check_and_load_model()

    async def evaluate(self, img: np.ndarray) -> np.ndarray:
        """Evaluate the model - find mask of the given image

        :param img: image to evaluate on
        :type img: np.ndarray
        :param z_axis: z dimension (optional, default is None)
        :type z_axis: int
        :return: mask of the image, list of 2D arrays, or single 3D array (if do_3D=True) labelled image.
        :rtype: np.ndarray
        """
        # update with the latest model if it is available (in case train has already occured)
        self.check_and_load_model()
        mask = self.model.eval(img=img)

        return mask

    def check_and_load_model(self) -> None:
        """Checks if the specified model exists in BentoML's model repository.
        If the model exists, it loads the latest version of the model into
        memory.
        """
        bento_model_list = [model.tag.name for model in bentoml.models.list()]
        if self.save_model_path in bento_model_list:
            loaded_model = bentoml.picklable_model.load_model(
                self.save_model_path + ":latest"
            )
            assert (
                loaded_model.__class__.__name__ == self.model.__class__.__name__
            ), "Check your config, loaded model and model to use not the same!"
            self.model = loaded_model
