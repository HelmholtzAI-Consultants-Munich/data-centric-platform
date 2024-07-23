# DCP Client
The client of our data centric platform for microscopy imaging.

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
[![Documentation Status](https://readthedocs.org/projects/data-centric-platform/badge/?version=latest)](https://data-centric-platform.readthedocs.io/en/latest/?badge=latest)

## How to use?

### Installation 

For installing dcp-client you will first need to clone the repo:

```
git clone https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform.git
```

Then navigate to the client directory:
```
cd data-centric-platform/src/client
```

In your dedicated environment run:
```
pip install -e .
```

This installation has been thoroughly tested using a conda environment with python version 3.9, 3.10, 3.11 and 3.12 on a macOS local machine.



#### Launch DCP client
Make sure the server is already running, either locally or remotely. Then, depending on the configuration, simply run:
```
dcp-client --mode local
```
or 
```
dcp-client --mode remote
```
depending on whether your server is running locally or remotely.

## Want to know more?
Visit our [documentation](https://data-centric-platform.readthedocs.io/en/latest/dcp_client_installation.html) for more information and a step by step guide on how to run the client.
