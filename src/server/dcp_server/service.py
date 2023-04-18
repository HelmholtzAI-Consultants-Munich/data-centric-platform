from __future__ import annotations
import bentoml
import typing as t
from fsimagestorage import FilesystemImageStorage
from bentomlrunners import CustomRunnable
from segmentationclasses import GeneralSegmentation
from serviceclasses import CustomBentoService
from utils import read_config

module = __import__("models")

# Import configuration
service_config = read_config('service', config_path = 'config.cfg')
model_config = read_config('model', config_path = 'config.cfg')
train_config = read_config('train', config_path = 'config.cfg')
eval_config = read_config('eval', config_path = 'config.cfg')

# Initiate the model
model_class = getattr(module, service_config['model_to_use'])
model = model_class(model_config = model_config, train_config = train_config, eval_config = eval_config)


custom_model_runner = t.cast(
    "CustomRunner", bentoml.Runner(CustomRunnable, name=service_config['runner_name'],
                                       runnable_init_params={"model": model, "save_model_path": service_config['save_model_path']})
)

# This is where we decide which segmentation we use - see segmentationclasses.py
# We also decide which filesystemimagestorage we use (currently there is one, as in client)
segmentation = GeneralSegmentation(imagestorage=FilesystemImageStorage(), 
                                   runner = custom_model_runner )

# Call the service
service = CustomBentoService(runner=segmentation.runner, segmentation=segmentation, service_name=service_config['service_name'])
svc = service.start_service()