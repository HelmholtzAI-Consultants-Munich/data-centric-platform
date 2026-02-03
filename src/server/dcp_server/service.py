from __future__ import annotations
import os
import sys
import numpy as np
import bentoml
import logging
import logging.handlers

from dcp_server.serviceclasses import CustomRunnable
from dcp_server.utils.fsimagestorage import FilesystemImageStorage
from dcp_server.utils.helpers import read_config
from dcp_server.utils.logger import get_logger

# Configure root logger IMMEDIATELY when service.py is loaded (in BentoML subprocess)
log_file = os.path.join(os.path.expanduser("~"), ".dcp_server", "dcp_server.log")
log_path = os.path.dirname(log_file)
os.makedirs(log_path, exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# File handler
file_handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)

# Add handlers only if not already present
if not root_logger.handlers:
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

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
            # Determine GPU usage for this inference
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except Exception:
                gpu_available = False

            runner_supports_gpu = False
            try:
                supports = getattr(seg.runner, "SUPPORTED_RESOURCES", None)
                if supports:
                    runner_supports_gpu = any(
                        ("gpu" in str(s).lower() or "cuda" in str(s).lower()) for s in supports
                    )
            except Exception:
                runner_supports_gpu = False

            model_on_cuda = False
            try:
                model_obj = getattr(seg, "model", None)
                
                # First check if model has device attribute (e.g., Cellpose models)
                device_attr = getattr(model_obj, "device", None)
                if device_attr and "cuda" in str(device_attr).lower():
                    model_on_cuda = True
                
                # Otherwise check parameters
                if not model_on_cuda:
                    params = getattr(model_obj, "parameters", None)
                    if callable(params):
                        for p in params():
                            if getattr(p, "is_cuda", False):
                                model_on_cuda = True
                                break
            except Exception:
                model_on_cuda = False

            using_gpu = model_on_cuda or (gpu_available and runner_supports_gpu)
            logger.info(
                f"GPU available={gpu_available}; runner_supports_gpu={runner_supports_gpu}; model_on_cuda={model_on_cuda}; using_gpu_for_inference={using_gpu}"
            )

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
