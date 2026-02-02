from __future__ import annotations
import os
import numpy as np
import bentoml

from dcp_server.serviceclasses import CustomRunnable
from dcp_server.utils.fsimagestorage import FilesystemImageStorage
from dcp_server.utils.helpers import read_config
from dcp_server.utils.logger import get_logger

logger = get_logger(__name__)

# -------------------------------
# 1. Load configs once, globally
# -------------------------------
logger.debug("Loading configuration files...")
script_path = os.path.abspath(__file__)
config_path = os.path.join(os.path.dirname(script_path), "config.yaml")
service_config = read_config("service", config_path=config_path)
model_config = read_config("model", config_path=config_path)
data_config = read_config("data", config_path=config_path)
eval_config = read_config("eval", config_path=config_path)
setup_config = read_config("setup", config_path=config_path)
logger.info("Configuration loaded successfully")

# -------------------------------
# 2. Instantiate model globally
# -------------------------------
logger.info(f"Initializing model: {setup_config.get('model_to_use', 'unknown')}")
from dcp_server.models import CustomCellpose
model = CustomCellpose(
    model_name=setup_config["model_to_use"],
    model_config=model_config,
    data_config=data_config,
    eval_config=eval_config,
)
logger.info("Model initialized successfully")

# -------------------------------
# 3. Create Runner globally
# -------------------------------
logger.debug(f"Creating runner: {service_config.get('runner_name', 'unknown')}")
runner = CustomRunnable(name=service_config["runner_name"],
                        model= model,
                        save_model_path= service_config["bento_model_path"]
)
logger.debug("Runner created successfully")

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
        logger.info(f"Setting up SegmentationService: {self.service_name}")
        segmentation_module = __import__("segmentationclasses")
        segm_class = getattr(segmentation_module, setup_config["segmentation"])
        fsimagestorage = FilesystemImageStorage(data_config, setup_config["model_to_use"])
        self.segmentation = segm_class(
            imagestorage=fsimagestorage,
            runner=runner,
            model=model,
        )
        logger.info(f"[SegmentationService] Initialized segmentation with service_name={self.service_name}")

    @bentoml.api
    async def segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segments a single pre-loaded image.
        
        :param image: Pre-loaded image as numpy array
        :type image: np.ndarray
        :return: Segmentation mask
        :rtype: np.ndarray
        """
        logger.debug(f"segment_image called with image shape={image.shape}")
        seg = self.segmentation
        
        try:
            # Prepare the image for segmentation
            prepared_img = seg.imagestorage.prepare_img_for_eval(image)
            
            # Add channel axis into the model's evaluation parameters dictionary
            seg.model.eval_config["segmentor"][
                "channel_axis"
            ] = seg.imagestorage.channel_ax
            
            # Evaluate the model
            mask = await seg.runner.evaluate(img=prepared_img)
            
            # Prepare the mask for saving
            mask = seg.imagestorage.prepare_mask_for_save(
                mask, seg.model.eval_config["mask_channel_axis"]
            )
            
            logger.debug(f"Segmentation complete for image with shape={image.shape}")
            return mask
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise
