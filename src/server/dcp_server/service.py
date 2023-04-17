from __future__ import annotations
import bentoml
import typing as t

from fsimagestorage import FilesystemImageStorage
from bentomlrunners import CustomRunnable
from segmentationclasses import GeneralSegmentation
from serviceclasses import OurBentoService
from models import CustomCellposeModel

# This is where we decide on the model type
model = CustomCellposeModel(model_type='cyto')

custom_model_runner = t.cast(
    "CustomRunner", bentoml.Runner(CustomRunnable, name="cellpose_runner",
                                       runnable_init_params={"model": model})
)

# This is where we decide which segmentation we use - see segmentationclasses.py
# We also decide which filesystemimagestorage we use (currently there is one, as in client)
segmentation = GeneralSegmentation(imagestorage=FilesystemImageStorage(), 
                                   runner = custom_model_runner )

# Call the service
service = OurBentoService(runner=segmentation.runner, segmentation=segmentation, service_name="cellpose_segm_test")
svc = service.start_service()