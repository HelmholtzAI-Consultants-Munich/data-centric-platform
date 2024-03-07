"""
Overview of dcp_server Package
==============================

The dcp_server package is structured to handle various server-side functionalities related to image processing, segmentation, and model serving. 

Submodules:
------------

dcp_server.fsimagestorage
    Provides a class FilesystemImageStorage for dealing with image storage, loading, saving, and processing.
    Contains methods for retrieving image-segmentation pairs, getting image size properties, loading images, preparing images and masks for training, rescaling images, resizing masks, saving images, and searching for images and segmentations in directories.

dcp_server.models
    Defines various models for cell classification and segmentation, including CellClassifierFCNN, CellClassifierShallowModel, CellposePatchCNN, CustomCellposeModel, and UNet.
    These models handle tasks such as evaluation, forward pass, training, and updating configurations.

dcp_server.segmentationclasses
    Defines segmentation classes for specific projects, such as GFPProjectSegmentation, GeneralSegmentation, and MitoProjectSegmentation.
    These classes contain methods for segmenting images and training models on images and masks.

dcp_server.serviceclasses
    Defines service classes, such as CustomBentoService and CustomRunnable, for serving the models with BentoML and handling computation on remote Python workers.

dcp_server.utils
    Provides various utility functions for image processing, feature extraction, file handling, configuration reading, and path manipulation.

"""
