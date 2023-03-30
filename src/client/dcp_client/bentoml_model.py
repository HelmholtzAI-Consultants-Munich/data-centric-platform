import asyncio
from bentoml.client import Client


class BentomlModel:

    def __init__(
        self,
    ):
        self.client = Client.from_url("http://0.0.0.0:7010") # have the url of the bentoml service here

    async def _run_train(self, data_path):
        response = await self.client.async_retrain(data_path)
        return response

    def run_train(self, data_path):
        return asyncio.run(self._run_train(data_path))

    async def _run_inference(self, data_path):
        response = await self.client.async_segment_image(data_path)
        return response
    
    def run_inference(self, data_path):
        list_of_files_not_suported = asyncio.run(self._run_inference(data_path))
        return list_of_files_not_suported