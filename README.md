# Data Centric Platform
*A data centric platform for all-kinds segmentation in microscopy imaging*

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
![tests](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/actions/workflows/test.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform)
[![Documentation Status](https://readthedocs.org/projects/data-centric-platform/badge/?version=latest)](https://data-centric-platform.readthedocs.io/en/latest/?badge=latest)


## How to use this?

This repo includes a client and server side for using our data centric platform. The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training. For full functionality of the software the server should be running, either locally or remotely. 

To install and start the server side, follow the instructions described in [DCP Server Installation & Launch](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md#using-pypi).

To run the client GUI follow the instructions described in [DCP Client Installation & Launch](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/README.md).

For an overview of both components and their interaction with the step-by-step guide and screen shots, visit our [documentation page](https://data-centric-platform.readthedocs.io/en/latest/index.html).

DCP handles all kinds of **segmentation tasks**! Try it out if you need to do:
* **Instance** segmentation
* **Semantic** segmentation
* **Multi-class instance** segmentation

### Toy data
This repo includes the ```data/``` directory with some toy data which you can use as the *Uncurated dataset* folder. You can create (empty) folders for the other two directories required in the welcome window and start playing around.

### Enabling data centric development
Our platform encourages the use of data centric practices. With the user friendly client interface you can:
- Detect and remove outliers from your training data: only confirmed samples are used to train our models
- Detect and correct labeling errors: editing labels with the integrated napari visualisation tool
- Establish consensus: allows for multiple annotators before curated label is passed to train model
- Focus on data curation: no interaction with model parameters during training and inference

#### *Get more with less!*
<img src="https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/readme_figs/dcp_pipeline.png"  width="200" height="200">
