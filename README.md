# data-centric-platform
A data centric platform for microscopy imaging

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/active-learning-platform)
![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)

## Installation
Simply run:
```
pip install -r requirements.txt
```

## How to use this?

This repo includes a client and server side for using our data centric platform. The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training. To start the server side you will need to run:

```
cd src/data-centric-platform
serve service:svc --reload --port=7010
```

To run the user interface (while the server is running) do: ```python client.py```

### Toy data
This repo includes the data/ directory with some toy data which you can use as the 'uncurated dataset' folder. You can create (empty) folders for the other two directories required in the welcome window.

