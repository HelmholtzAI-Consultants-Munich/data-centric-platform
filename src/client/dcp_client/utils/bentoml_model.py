from typing import Optional, List
from bentoml import SyncHTTPClient
from bentoml.exceptions import BentoMLException
import numpy as np
from numpy.typing import NDArray

from dcp_client.app import Model
from dcp_client.utils.logger import get_logger

logger = get_logger(__name__)



class BentomlModel(Model):
    """BentomlModel class for connecting to a BentoML server and running inference tasks."""

    def __init__(self, client: Optional[SyncHTTPClient] = None):
        """Initializes the BentomlModel.

        :param client: Optional SyncHTTPClient instance. If None, it will be initialized during connection.
        :type client: Optional[SyncHTTPClient]
        """
        self.client = client

    def connect(self, ip: str = "0.0.0.0", port: int = 7010) -> bool:
        """Connects to the BentoML server.

        :param ip: IP address of the BentoML server. Default is '0.0.0.0'.
        :type ip: str
        :param port: Port number of the BentoML server. Default is 7010.
        :type port: int
        :return: True if connection is successful, False otherwise.
        :rtype: bool
        """
        url = f"http://{ip}:{port}"  # "http://0.0.0.0:7010"
        try:
            logger.info(f"Attempting to connect to BentoML server at {url}")
            self.client = SyncHTTPClient(url)
            logger.info(f"Successfully connected to BentoML server at {url}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to BentoML server at {url}: {e}")
            return False  # except ConnectionRefusedError

    @property
    def is_connected(self) -> bool:
        """Checks if the BentomlModel is connected to the BentoML server.

        :return: True if connected, False otherwise.
        :rtype: bool
        """
        return bool(self.client)

    async def segment_image(self, image: NDArray) -> NDArray:
        """Segments a single image.
        
        :param image: Pre-loaded image as numpy array
        :type image: NDArray
        :return: Segmentation mask
        :rtype: NDArray
        """
        try:
            logger.debug("Running inference on single image")
            response = self.client.segment_image(image)
            logger.debug("Inference completed for image")
            return response
        except BentoMLException as e:
            logger.error(f"BentoML error during inference: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during inference: {e}")
            raise

    def run_train(self, path: str) -> None:
        """Training functionality has been removed from the server.
        
        :param path: Path to training data (not used).
        :type path: str
        :raises NotImplementedError: Training is no longer available.
        """
        raise NotImplementedError("Training functionality has been removed from the server. Please use inference only.")
