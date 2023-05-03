from typing import List
import bentoml
import numpy as np


class CustomRunnable(bentoml.Runnable):
    '''
    BentoML, Runner represents a unit of computation that can be executed on a remote Python worker and scales independently.
    CustomRunnable is a custom runner defined to meet all the requirements needed for this project.
    '''
    SUPPORTED_RESOURCES = ("cpu",) #TODO add here?
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(self, model, save_model_path):
        """Constructs all the necessary attributes for the CustomRunnable.

        :param model: model to be trained or evaluated
        :param save_model_path: full path of the model object that it will be saved into
        :type save_model_path: str
        """        
        
        self.model = model
        self.save_model_path = save_model_path

    @bentoml.Runnable.method(batchable=False)
    def evaluate(self, img: np.ndarray, **eval_config) -> np.ndarray:
        """Evaluate the model - find mask of the given image

        :param img: image to evaluate on
        :type img: np.ndarray
        :param z_axis: z dimension (optional, default is None)
        :type z_axis: int
        :return: mask of the image, list of 2D arrays, or single 3D array (if do_3D=True) labelled image.
        :rtype: np.ndarray
        """              

        mask = self.model.eval(img=img, **eval_config)

        return mask

    @bentoml.Runnable.method(batchable=False)
    def train(self, imgs: List[np.ndarray], masks: List[np.ndarray]) -> str:
        """Trains the given model

        :param imgs: images to train on (training data)
        :type imgs: List[np.ndarray]
        :param masks: masks of the given images (training labels)
        :type masks: List[np.ndarray]
        :return: path of the saved model
        :rtype: str
        """        

        self.model.train(imgs, masks)

        # Save the bentoml model
        bentoml.picklable_model.save_model(self.save_model_path, self.model) 

        return self.save_model_path