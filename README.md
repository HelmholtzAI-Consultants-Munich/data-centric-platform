# Data Centric Platform
*A data centric platform for mutli-class segmentation in microscopy imaging*

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/active-learning-platform)

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
[![tests](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/workflows/tests/badge.svg)](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/actions)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform)



## How to use this?

This repo includes a client and server side for using our data centric platform. The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training. For full functionality of the software the server should be running, either locally or remotely. To install and start the server side follow the instructions described in [DCP Server Installation & Launch](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md#using-pypi).

To run the client GUI follow the instructions described in [DCP Client Installation & Launch](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/README.md).

### Toy data
This repo includes the ```data/``` directory with some toy data which you can use as the *Uncurated dataset* folder. You can create (empty) folders for the other two directories required in the welcome window and start playing around.
