import asyncio
from typing import Optional

from bentoml.client import Client as BentoClient

accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")


class MLClient:
    def __init__(self, client: Optional[BentoClient] = None) -> None:
        self.client = client
        
    def connect(self, ip: str = '0.0.0.0', port: int = 7010) -> None:
        url = f"http://{ip}:{port}"
        self.client = BentoClient.from_url(url) # have the url of the bentoml service here

    @property
    def is_connected(self) -> bool:
        return bool(self.client)

    # run_train
    def run_train(self, path) -> str:
        return asyncio.run(self._run_train(path))
    
    async def _run_train(self, path):
        response = await self.client.async_retrain(path)
        return response

    # run_inference
    def run_inference(self, path) -> str:
        list_of_files_not_suported = asyncio.run(self._run_inference(path))
        list_of_files_not_suported = list(list_of_files_not_suported)
        if len(list_of_files_not_suported) > 0:
            return "Image types not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
            Currently supported image file formats are: ", accepted_types, "The files that were not supported are: " + ", ".join(list_of_files_not_suported)
        else:
            return "Success! Masks generated for all images"
    

    async def _run_inference(self, path):
        response = await self.client.async_segment_image(path)
        return response
    