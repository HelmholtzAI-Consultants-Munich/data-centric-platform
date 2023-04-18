from __future__ import annotations
import bentoml
import typing as t
from fsimagestorage import FilesystemImageStorage
from bentomlrunners import CustomRunnable
from segmentationclasses import GeneralSegmentation
from serviceclasses import OurBentoService
from models import CustomCellposeModel
from utils import read_config

# Import configuration
service_config = read_config('service', config_path = 'config.cfg')
model_config = read_config('model', config_path = 'config.cfg')
train_config = read_config('train', config_path = 'config.cfg')
eval_config = read_config('eval', config_path = 'config.cfg')

# Initiate the model
model = CustomCellposeModel(model_type=model_config['model_type'], train_config = train_config, eval_config = eval_config)


custom_model_runner = t.cast(
    "CustomRunner", bentoml.Runner(CustomRunnable, name="cellpose_runner",
                                       runnable_init_params={"model": model, "save_model_path": service_config['save_model_path']})
)

# This is where we decide which segmentation we use - see segmentationclasses.py
# We also decide which filesystemimagestorage we use (currently there is one, as in client)
segmentation = GeneralSegmentation(imagestorage=FilesystemImageStorage(), 
                                   runner = custom_model_runner )

# Call the service
service = OurBentoService(runner=segmentation.runner, segmentation=segmentation, service_name="cellpose_segm_test")
svc = service.start_service()