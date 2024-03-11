# DCP Client
The client of our data centric platform for microscopy imaging.

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
[![Documentation Status](https://data-centric-platform.readthedocs.io/en/latest/dcp_client_installation.html)](https://data-centric-platform.readthedocs.io/en/latest/?badge=latest)

## How to use?

### Installation
This has been tested on Python versions 3.9, 3.10 and 3.11 on latest versions of Windows, Ubuntu and MacOS. In your dedicated environment run:
```
pip install dcp_client
```

### Installation for developers
Before starting, make sure you have navigated to ```data-centric-platform/src/client```. All future steps expect you are in the client directory. This installation has been tested using a conda environment with python version 3.9 on a mac local machine. In your dedicated environment run:
```
pip install -e .
```

#### Launch DCP client
Make sure the server is already running, either locally or remotely. Then, depending on the configuration, simply run:
```
dcp-client --mode local/remote
```

## Want to know more?
Visit our [documentation](https://data-centric-platform.readthedocs.io/en/latest/dcp_client_installation.html) for more information and a step by step guide on how to run the client.
