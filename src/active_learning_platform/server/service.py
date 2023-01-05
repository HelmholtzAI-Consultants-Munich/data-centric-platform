import typing as t
from PIL.Image import Image as PILImage
from numpy.typing import NDArray
import numpy as np

import bentoml
from bentoml.io import Image
from bentoml.io import NumpyNdarray
# from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

# Retrieve the latest model
mnist_runner = bentoml.pytorch.get("pytorch_mnist:latest").to_runner()
svc = bentoml.Service(name="poc-demo", runners=[mnist_runner])
# svc.add_asgi_middleware(HTTPSRedirectMiddleware)

@svc.api(input=Image(), output=NumpyNdarray(dtype="int64"))
async def predict(f: PILImage):
    print("Calling predict_image from server.")
    assert isinstance(f, PILImage)
    arr = np.array(f) / 255.0
    assert arr.shape == (28, 28)
    arr = np.expand_dims(arr, (0, 1)).astype("float32")
    output_tensor = await mnist_runner.async_run(arr)
    output_arr = output_tensor.detach().cpu().numpy()
    return output_arr


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def retrain(arr):
    print("Calling retrain from server.")
    arr = np.expand_dims(arr, (1)).astype("float32")
    return arr