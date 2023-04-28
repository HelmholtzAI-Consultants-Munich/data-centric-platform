from __future__ import annotations
import numpy as np
import bentoml
from bentoml.io import Text, NumpyNdarray


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
    
