.. _DCP Server:

DCP Server
===========


The server of our data centric platform for microscopy imaging.

.. image:: https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg
   :alt: stability-wip

The client and server communicate via the `bentoml <https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB>`_ library. The client interacts with the server every time we run model inference or training, so the server should be running before starting the client.


Installation
--------------

Before starting make sure you have navigated to ``data-centric-platform/src/server``. All future steps expect you are in the server directory. In your dedicated environment run:

.. code-block:: bash

   pip install -e .

Launch DCP Server
------------------

Simply run:

.. code-block:: bash

   python dcp_server/main.py

Once the server is running, you can verify it is working by visiting http://localhost:7010/ in your web browser.

Customization (for developers)
--------------------------------

All service configurations are set in the ``config.cfg`` file. Please, obey the `formal JSON format <https://www.json.org/json-en.html>`_.

The config file has to have the six main parts. All the ``marked`` arguments are mandatory:

- ``setup``

  - ``segmentation`` - segmentation type from the ``segmentationclasses.py``. Currently, only ``GeneralSegmentation`` is available (MitoProjectSegmentation and GFPProjectSegmentation are stale).
  - ``model_to_use`` - name of the model class from the ``models.py`` you want to use. Currently, available models are: ``CustomCellposeModel``, ``UNet`` and ``CellposePatchCNN``. See **Models** section for more information. 
  - ``accepted_types`` - types of images currently accepted for the analysis
  - ``seg_name_string`` - end string for masks to run on (All the segmentations of the image should contain this string - used to save and search for segmentations of the images)
- ``service``

  - ``runner_name`` - name of the runner for the bentoml service
  - ``bento_model_path`` - name for the trained model which will be saved after calling the (re)train from service - is saved under ``bentoml/models``
  - ``service_name`` - name for the bentoml service
  - ``port`` - on which port to start the service
- ``model`` - configuration for the model instantiation. Here, pass any arguments you need or want to change. Take care that the names of the arguments are the same as of the original model class ``__init__()`` function.

  - ``segmentor``: model configuration for the segmentor. ``CustomCellposeModel`` takes arguments used in the init of CellposeModel, see `here <https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel>`__.
  - ``classifier``: model configuration for classifier, see ``__init__()`` of ``UNet``, ``CellClassifierFCNN`` or ``CellClassifierShallowModel``.
- ``data`` - data configuration

  - ``data_root``: if you are running the server remotely, then you need to specify the project path here. Should match the ``server: data-path`` argument in the client config.
- ``train`` - configuration for the model training. Take care that the names of the arguments are the same as of the original model's ``train()`` function.
  
  - ``segmentor``: for  ``CustomCellposeModel`` the ``train()`` function arguments can be found `here <https://cellpose.readthedocs.io/en/latest/api.html#id7>`__. Pass any arguments you need or want to change or leave empty {}, then default arguments will be used.
  - ``classifier``: train configuration for classifier, see ``train()`` of ``UNet``, ``CellClassifierFCNN`` or ``CellClassifierShallowModel``.
- ``eval`` - configuration for the model evaluation. Take care that the names of the arguments are the same as of the original model's ``eval()`` function.
  
  - ``segmentor``: for  ``CustomCellposeModel`` the ``eval()`` function arguments can be found `here <https://cellpose.readthedocs.io/en/latest/api.html#id3>`__. Pass any arguments you need or want to change or leave empty {}, then default arguments will be used.
  - ``classifier``: train configuration for classifier, see ``eval()`` of ``UNet``, ``CellClassifierFCNN``or ``CellClassifierShallowModel``
  - ``mask_channel_axis``: If a multi-class instance segmentation model has been used, then the masks returned by the model should have two channels, one for the instance segmentation results and one indicating the objects class. This variable indicated at which dim the channel axis should be stored. Currently should be kept at 0, as this is the only way the masks can be visualized correctly by napari in the client.

To make it easier to run the server we provide you with three config files: 
 
 - ``config.cfg`` is set up to work for a panoptic segmentation task
 - ``config_instance.cfg`` for instance segmentation
 - ``config_semantic.cfg`` for semantic segmentation
Make sure to rename the config you wish to use to ``config.cfg``. The default is panoptic segmentation.

Models
-------

The models are currently integrated into DCP:

- **Instance** Segmentation: 

  - ``CustomCellposeModel``: Inherits from cellpose.models.CellposeModel, see `here <https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel>`__ for more information.
- **Semantic** Segmentation: 
  
  - ``UNet``: A vanilla U-Net model, trained on the full images
- **Panoptic** Segmentation: 

  - ``CellposePatchCNN``: Includes a segmentor for instance segmentation, sequentially followed by a classifier for semantic segmentation. The segmentor can only be ``CustomCellposeModel`` model, while the classifier can be one of:

    - ``CellClassifierFCNN`` or "FCNN" (in config): A CNN model for obtaining class labels, trained on images patches of individual objects, extarcted using the instance mask from the previous step
    - ``CellClassifierShallowModel`` or "RandomForest" (in config): A Random Forest model for obtaining class labels, trained on shape and intensity features of the objects, extracted using the instance mask from the previous step.
  - UNet: If the post-processing argument is set, then the instance mask is deduced from the labels mask. Will not be able to handle touching objects 


Running with Docker 
-------------------------------------------------------

.. note::
    DO NOT USE UNTIL ISSUE IS SOLVED: Currently doesn't work for generate labels

Docker-Compose
~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker compose up

Docker Non-Interactively
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker build -t dcp-server .
   docker run -p 7010:7010 -it dcp-server

Docker Interactively
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   docker build -t dcp-server .
   docker run -it dcp-server bash
   bentoml serve service:svc --reload --port=7010

