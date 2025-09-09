from typing import Optional, List
from bentoml import SyncHTTPClient
from bentoml.exceptions import BentoMLException
import numpy as np

from dcp_client.app import Model


class BentomlModel(Model):
    """BentomlModel class for connecting to a BentoML server and running training and inference tasks."""

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
            self.client = SyncHTTPClient(url)
            return True
        except:
            return False  # except ConnectionRefusedError

    @property
    def is_connected(self) -> bool:
        """Checks if the BentomlModel is connected to the BentoML server.

        :return: True if connected, False otherwise.
        :rtype: bool
        """
        return bool(self.client)

    def _run_train(self, data_path: str) -> Optional[str]:
        """Runs the training task asynchronously.

        :param data_path: Path to the training data.
        :type data_path: str
        :return: Response from the server if successful, None otherwise.
        :rtype: str, or None
        """
        try:
            response = self.client.train(data_path) # train is part of running server
            return response
        except BentoMLException:
            return None

    def run_train(self, data_path: str):
        """Runs the training.

        :param data_path: Path to the training data.
        :type data_path: str
        :return: Response from the server if successful, None otherwise.
        """
        return self._run_train(data_path)

    def _run_inference(self, data_path: str) -> Optional[np.ndarray]:
        """Runs the inference task asynchronously.

        :param data_path: Path to the data for inference.
        :type data_path: str
        :return: List of files not supported by the server if unsuccessful, otherwise returns None.
        :rtype: np.ndarray, or None
        """
        try:
            response = self.client.segment_image(data_path) # segment_image is part of running server
            return response
        except BentoMLException:
            return None

    def run_inference(self, data_path: str) -> List:
        """Runs the inference.

        :param data_path: Path to the data for inference.
        :type data_path: str
        :return: List of files not supported by the server if unsuccessful, otherwise returns None.
        """
        list_of_files_not_suported = self._run_inference(data_path)
        return list_of_files_not_suported
