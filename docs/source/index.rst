.. dcp documentation master file, created by
   sphinx-quickstart on Sun Feb 11 18:53:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Data Centric Platform
===============================

*A data centric platform for all-kinds segmentation in microscopy imaging*

.. image:: https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg
   :alt: stability-wip

.. image:: https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/actions/workflows/test.yml/badge.svg?event=push
   :alt: tests

.. image:: https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform

How to use it?
----------------

The client and server communicate via the `bentoml <https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB>`_ library. 
The client interacts with the server every time we run model inference or training. 
For full functionality of the software the server should be running, either locally or remotely. 

To install and start the server side follow the instructions described in :ref:`DCP Server`.
To run the client GUI follow the instructions described in :ref:`DCP Client`

DCP handles all kinds of **segmentation tasks**! Try it out if you need to do:
- **Instance** segmentation
- **Semantic** segmentation
- **Panoptic** segmentation

Toy data
--------

Our github repo includes the ``data/`` directory with some toy data which you can use as the *Uncurated dataset* folder. You can create (empty) folders for the other two directories required in the welcome window and start playing around.

Enabling data centric development
----------------------------------

Our platform encourages the use of data centric practices. With the user friendly client interface you can:
- Detect and remove outliers from your training data: only confirmed samples are used to train our models
- Detect and correct labeling errors: editing labels with the integrated napari visualisation tool
- Establish consensus: allows for multiple annotators before curated label is passed to train model
- Focus on data curation: no interaction with model parameters during training and inference


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   dcp_client_installation
   dcp_server_installation
   dcp_server
   dcp_client

.. centered::
      *Get more with less!*