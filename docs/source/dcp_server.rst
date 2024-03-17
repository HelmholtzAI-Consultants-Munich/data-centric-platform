dcp\_server package
===================

The dcp_server package is structured to handle various server-side functionalities related model serving for segmentation and training. 

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


Submodules
----------

dcp\_server.models module
-------------------------

.. automodule:: dcp_server.models
   :members:
   :undoc-members:
   :show-inheritance:

dcp\_server.segmentationclasses module
--------------------------------------

.. automodule:: dcp_server.segmentationclasses
   :members:
   :undoc-members:
   :show-inheritance:

dcp\_server.serviceclasses module
---------------------------------

.. automodule:: dcp_server.serviceclasses
   :members:
   :undoc-members:
   :show-inheritance:

dcp\_server.utils module
---------------------------------

.. toctree::
   :maxdepth: 4

   dcp_server.utils