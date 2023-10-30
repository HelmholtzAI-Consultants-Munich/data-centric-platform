from __future__ import annotations
import numpy as np
import bentoml
from bentoml.io import Text, NumpyNdarray
from typing import List


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
    def evaluate(self, img: np.ndarray) -> np.ndarray:
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
        s1 = self.model.segmentor.net.state_dict()
        c1 = self.model.classifier.parameters()
        self.model.train(imgs, masks)
        s2 = self.model.segmentor.net.state_dict()
        c2 = self.model.classifier.parameters()
        if s1 == s2: print('S1 and S2 COMP: THEY ARE THE SAME!!!!!')
        else: print('S1 and S2 COMP: THEY ARE NOOOT THE SAME!!!!!')
        for p1, p2 in zip(c1, c2):
            if p1.data.ne(p2.data).sum() > 0:
                print("C1 and C2 NOT THE SAME")
                break
        # Save the bentoml model
        bentoml.picklable_model.save_model(self.save_model_path, self.model) 

        return self.save_model_path
    

class CustomBentoService():
    """BentoML Service class. Contains all the functions necessary to serve the service with BentoML
    """    
    def __init__(self, runner, segmentation, service_name):
        """Constructs all the necessary attributes for the class CustomBentoService():

        :param runner: runner used in the service
        :type runner: CustomRunnable class object
        :param segmentation: segmentation type used in the service
        :type segmentation: segmentation class object from the segmentationclasses.py
        :param service_name: name of the service 
        :type service_name: str
        """        
        self.runner = runner
        self.segmentation = segmentation
        self.service_name = service_name

    def start_service(self):
        """Starts the service

        :return: service object needed in service.py and for the bentoml serve call.
        """        
        svc = bentoml.Service(self.service_name, runners=[self.runner])

        @svc.api(input=Text(), output=NumpyNdarray())  #input path to the image output message with success and the save path
        async def segment_image(input_path: str):
            """function served within the service, used to segment images

            :param input_path: directory where the images for segmentation are saved
            :type input_path: str
            :return: list of files not supported
            :rtype: ndarray
            """            
            list_of_images = self.segmentation.imagestorage.search_images(input_path)
            list_of_files_not_suported = self.segmentation.imagestorage.get_unsupported_files(input_path)
    
            if not list_of_images:
                return np.array(list_of_images)
            else:
                await self.segmentation.segment_image(input_path, list_of_images)

            return np.array(list_of_files_not_suported)

        @svc.api(input=Text(), output=Text())
        async def train(input_path):
            """function served within the service, used to retrain the model

            :param input_path: directory where the images for training are saved
            :type input_path: str
            :return: message of success if training went well
            :rtype: str
            """            
            print("Calling retrain from server.")

            # Train the model
            model_path = await self.segmentation.train(input_path)

            msg = "Success! Trained model saved in: " + model_path

            return msg
        
        return svc
    
