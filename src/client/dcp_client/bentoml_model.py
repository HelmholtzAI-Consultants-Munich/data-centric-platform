import asyncio
from typing import Optional
from bentoml.client import Client as BentoClient

from dcp_client.app import Model

class BentomlModel(Model):

    def __init__(
        self,
        client: Optional[BentoClient] = None
    ):
        self.client = client

    def connect(self, ip: str = '0.0.0.0', port: int = 7010):
        url = f"http://{ip}:{port}" #"http://0.0.0.0:7010"
        try:
            self.client = BentoClient.from_url(url) 
            return True
        except : return False # except ConnectionRefusedError
    
    @property
    def is_connected(self):
        return bool(self.client)

    async def _run_train(self, data_path):
        response = await self.client.async_train(data_path)
        return response

    def run_train(self, data_path):
        return asyncio.run(self._run_train(data_path))

    async def _run_inference(self, data_path):
        response = await self.client.async_segment_image(data_path)
        return response
    
    def run_inference(self, data_path):
        list_of_files_not_suported = asyncio.run(self._run_inference(data_path))
        return list_of_files_not_suported