# data-centric-platform
A data centric platform for microscopy imaging

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/active-learning-platform)
![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)



## How to use this?

This repo includes a client and server side for using our data centric platform. The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training. For full functionality of the software the server should be running. To start the server side you will need to run:

```
conda activate dcp-env
cd src/server/dcp_server
bentoml serve service:svc --reload --port=7010
```

To run the client GUI open a new terminal and do: 
```
conda activate dcp-env
python src/client/dcp_client/main.py
```

### Toy data
This repo includes the ```data/``` directory with some toy data which you can use as the 'uncurated dataset' folder. You can create (empty) folders for the other two directories required in the welcome window.

## Customization (for developers)

All service configurations are set in the _config.cfg_ file. Please, obey the [formal JSON format](https://www.json.org/json-en.html).

The config file has to have the five main parts. All the ```marked``` arguments are mandatory:

 - ``` setup ``` 
    - ```accepted_types``` - types of images currently accepted for the analysis
    - ```seg_name_string``` - end string for masks to run on (All the segmentations of the image should contain this string - used to save and search for segmentations of the images)
- ```service```
    - ```model_to_use``` - name of the model class from the models.py you want to use. Currently, only the CustomCellposeModel is available. 
    - ```save_model_path``` - path and and name for the trained model which will be saved after calling the (re)train from service
    - ```runner_name``` -  name of the runner for the bentoml service 
    - ```service_name``` - name for the service
- ```model``` - configuration for the model instatiation. Here, pass any arguments you need or want to change. Take care that the names of the arguments are the same as of original model class' _init()_ function!
- ```train``` - configuration for the model training. Here, pass any arguments you need or want to change or leave empty {}. Take care that the names of the arguments are the same as of original model's _train()_ function! If using cellpose - the _train()_ function arguments can be found [here](https://cellpose.readthedocs.io/en/latest/api.html#id7)
- ```eval``` - configuration for the model evaluation. Here, pass any arguments you need or want to change or leave empty {}. Take care that the names of the arguments are the same as of original model's _eval()_ function! If using cellpose - the _eval()_ function arguments can be found [here](https://cellpose.readthedocs.io/en/latest/api.html#id3).

