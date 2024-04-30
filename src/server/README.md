# DCP Server

The server of our data centric platform for microscopy imaging.

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
[![Documentation Status](https://readthedocs.org/projects/data-centric-platform/badge/?version=latest)](https://data-centric-platform.readthedocs.io/en/latest/?badge=latest)

The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training, so the server should be running before starting the client.

## How to use?

### Installation
This has been tested on Python versions 3.9, 3.10 and 3.11 on latest versions of Windows, Ubuntu and MacOS. In your dedicated environment run:
```
pip install dcp_server
```

### Installation for developers
Before starting make sure you have navigated to ```data-centric-platform/src/server```. All future steps expect you are in the server directory. You will need to have Python <3.12. In your dedicated environment run:
```
pip install numpy # if you don't do this, pyradiomics fails
pip install -e .
```

#### Launch DCP server
Simply run:
```
python dcp_server/main.py
```
Once the server is running, you can verify it is working by visiting http://localhost:7010/ in your web browser.

## Want to know more?
Visit our [documentation](https://data-centric-platform.readthedocs.io/en/latest/dcp_server_installation.html) for more information on server configurations and available models.
