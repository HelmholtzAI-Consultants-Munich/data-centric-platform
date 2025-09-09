from __future__ import annotations
import os
import numpy as np
import bentoml

from dcp_server.serviceclasses import CustomRunnable
from dcp_server.utils.fsimagestorage import FilesystemImageStorage
from dcp_server.utils.helpers import read_config

# -------------------------------
# 1. Load configs once, globally
# -------------------------------
script_path = os.path.abspath(__file__)
config_path = os.path.join(os.path.dirname(script_path), "config.yaml")

service_config = read_config("service", config_path=config_path)
model_config = read_config("model", config_path=config_path)
data_config = read_config("data", config_path=config_path)
train_config = read_config("train", config_path=config_path)
eval_config = read_config("eval", config_path=config_path)
setup_config = read_config("setup", config_path=config_path)

# -------------------------------
# 2. Instantiate model globally
# -------------------------------
models_module = __import__("models")
model_class = getattr(models_module, setup_config["model_to_use"])
model = model_class(
    model_name=setup_config["model_to_use"],
    model_config=model_config,
    data_config=data_config,
    train_config=train_config,
    eval_config=eval_config,
)

# -------------------------------
# 3. Create Runner globally
# -------------------------------
runner = bentoml.Runner(CustomRunnable,
                        name=service_config["runner_name"],
                        runnable_init_params={
                            "model": model,
                            "save_model_path": service_config["bento_model_path"],
                        },
)

# -------------------------------
# 4. Service Definition
# -------------------------------
@bentoml.service(runners=[runner])
class SegmentationService:
    """BentoML Service class. Contains all the functions necessary to serve the service with BentoML"""

    service_name = os.getenv("SERVICE_NAME", "default_segmentation_service")
    segmentation = None  # will be initialized at startup

    @bentoml.on_startup
    async def setup(self):
        # Instantiate segmentation dynamically at startup
        segmentation_module = __import__("segmentationclasses")
        segm_class = getattr(segmentation_module, setup_config["segmentation"])
        fsimagestorage = FilesystemImageStorage(data_config, setup_config["model_to_use"])
        self.segmentation = segm_class(
            imagestorage=fsimagestorage,
            runner=runner,
            model=model,
        )
        print(f"[SegmentationService] Initialized segmentation with service_name={self.service_name}")

    @bentoml.api
    async def segment_image(self, input_path: str) -> np.ndarray:
        seg = self.segmentation
        images = seg.imagestorage.search_images(input_path)
        unsupported = seg.imagestorage.get_unsupported_files(input_path)

        if not images:
            return np.array(images)
        await seg.segment_image(input_path, images)
        return np.array(unsupported)

    @bentoml.api
    async def train(self, input_path: str) -> str:
        seg = self.segmentation
        print("Calling retrain from server.")
        msg = await seg.train(input_path)
        if msg != seg.no_files_msg:
            return f"Success! Trained model saved in: {msg}"
        return msg
