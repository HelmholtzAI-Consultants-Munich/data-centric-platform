"""
Overview of dcp_server Package
==============================

The dcp_server package is structured to handle various server-side functionalities related model serving for segmentation and training. 

Submodules:
------------

dcp_server.models
    Defines various models for cell classification and segmentation, including CellClassifierFCNN, CellClassifierShallowModel, CellposePatchCNN, CustomCellposeModel, and UNet.
    These models handle tasks such as evaluation, forward pass, training, and updating configurations.

dcp_server.segmentationclasses
    Defines segmentation classes for specific projects, such as GFPProjectSegmentation, GeneralSegmentation, and MitoProjectSegmentation.
    These classes contain methods for segmenting images and training models on images and masks.

dcp_server.serviceclasses
    Defines service classes, such as CustomBentoService and CustomRunnable, for serving the models with BentoML and handling computation on remote Python workers.

dcp_server.utils
    Provides various utility functions for dealing with image storage, image processing, feature extraction, file handling, configuration reading, and path manipulation.

"""
