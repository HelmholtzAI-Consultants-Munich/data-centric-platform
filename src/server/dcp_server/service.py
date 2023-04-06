from __future__ import annotations
import bentoml
import typing as t

from fsimagestorage import FilesystemImageStorage
from bentomlrunners import CellposeRunnable
from segmentationclasses import GeneralSegmentation
from service_types import OurBentoService
from abc import ABC, abstractmethod


#TODO: do we want to put this into settings.py so we decide everything there?
# This is where we decide on the model type
custom_model_runner = t.cast(
    "CellposeRunner", bentoml.Runner(CellposeRunnable, name="cellpose_runner",
                                       runnable_init_params={"cellpose_model_type": "cyto"})
)

# This is where we decide which segmentation we use - see segmentationclasses.py
# We also decide which filesystemimagestorage we use (currently there is one, as in client)
segmentation = GeneralSegmentation(imagestorage=FilesystemImageStorage(), 
                                   runner = custom_model_runner )

# Call the service
service = OurBentoService(runner=segmentation.runner, segmentation=segmentation, service_name="cellpose_segm_test")
svc = service.start_service()