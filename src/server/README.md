# DCP Server

The server of our data centric platform for microscopy imaging.

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)

The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training, so the server should be running before starting the client.

## How to use?

### Installation
Before starting make sure you have navigated to ```data-centric-platform/src/server```. All future steps expect you are in the server directory. In your dedicated environment run:
```
pip install -e .
```

#### Launch DCP server
Simply run:
```
python dcp_server/main.py
```
Once the server is running, you can verify Models
The current models are currently integrated into DCP:

CellPose --> for instance segmentation tasks
CellposePatchCNN --> for panoptic segmentation tasks: includes the Cellpose model for instance segmentation followed by a patch wise CNN model on the predicted instances for obtaining class labels it is working by visiting http://localhost:7010/ in your web browser.


## Customization (for developers)

All service configurations are set in the _config.cfg_ file. Please, obey the [formal JSON format](https://www.json.org/json-en.html).

The config file has to have the five main parts. All the ```marked``` arguments are mandatory:

### Setup

- **`segmentation`**
  - Segmentation type from `segmentationclasses.py`. Currently, only **GeneralSegmentation** is available (MitoProjectSegmentation and GFPProjectSegmentation are stale).
  
- **`accepted_types`**
  - Types of images currently accepted for analysis.
  
- **`seg_name_string`**
  - End string for masks to run on. All segmentations of the image should contain this string - used to save and search for segmentations of the images.

### Service

- **`model_to_use`**
  - Name of the model class from `models.py` you want to use. Currently available models:
    - **CustomCellposeModel**: Inherits [CellposeModel](https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel) class
    - **CellposePatchCNN**: Includes a segmentor and a classifier. Currently, the segmentor can only be `CustomCellposeModel`, and the classifier can be either ``CellClassifierFCNN`` or ``CellClassifierShallowModel`` (Random Forest). The model sequentially runs the segmentor and then classifier on patches of the objects to classify them.
    - **UNet**: End-to-End segmentation model.
    - **CellposeMultichannel**:  Multichannel image segmentation model. Employs distinct CustomCellposeModel instances for each channel, the number of channels corresponds to the `num_classes` parameter of the `classifier`.

- **`save_model_path`**
  - Name for the trained model, which will be saved after calling the (re)train from service - saved under `bentoml/models`.
  
- **`runner_name`**
  - Name of the runner for the BentoML service.
  
- **`service_name`**
  - Name for the BentoML service.
  
- **`port`**
  - On which port to start the service.
 
### Model

- Configuration for the model instantiation. Pass any arguments you need or want to change. Ensure that the names of the arguments are the same as the original model class' `__init__()`!
  
  - **`segmentor`**: Model configuration for the segmentor. Currently takes arguments used in the init of CellposeModel, see [here](https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel).
  
  - **`classifier`**: Model configuration for the classifier. The type of classifier is determined by the ``model_class`` specified in the config. Available options:
    - If ``model_class`` is set to **"FCNN"**, see `__init__()` of `CellClassifierFCNN`.
    - If ``model_class`` is set to **"RandomForest"**, see `__init__()` of `CellClassifierShallowModel`.

For **UNet** and **CellposeMultichannel** see `__init__()` of the corresponding class.

### Train

- Configuration for the model training. Ensure that the names of the arguments are the same as the original model's `train()` function!
  
  - **segmentor**: If using Cellpose, the `train()` function arguments can be found [here](https://cellpose.readthedocs.io/en/latest/api.html#id7). Pass any arguments you need or want to change or leave empty `{}`, then default arguments will be used.
  
  - **classifier**: Train configuration for the classifier, see _train()_  of `CellClassifierFCNN`, `CellClassifierShallowModel` or `UNet`.


### Evaluation

- Configuration for the model evaluation. Ensure that the names of the arguments are the same as the original model's _eval()_ function!
  
  - **segmentor**: If using Cellpose, the _eval()_ function arguments can be found [here](https://cellpose.readthedocs.io/en/latest/api.html#id3). Pass any arguments you need or want to change or leave empty `{}`, then default arguments will be used.
  
  - **classifier**: Train configuration for the classifier, see _eval()_ of `CellClassifierFCNN`.
  
  - **mask_channel_axis**: If a multi-class instance segmentation model has been used, then the masks returned by the model should have two channels, one for the instance segmentation results and one indicating the objects class. This variable indicates at which dimension the channel axis should be stored. Currently should be kept at `0`, as this is the only way the masks can be visualized correctly by Napari in the client.

To make it easier for you we provide you with two config files: ```config.cfg``` is set up to work for a panoptic segmentation task, while ```config_instance.cfg``` for instance segmentation. Make sure to rename the config you wish to use to ```config.cfg```. The default is panoptic segmentation. 

## Models

The following models are currently integrated into DCP:

- **CellPose**: Used for instance segmentation tasks.

- **Cellpose+Classifier**: Designed for panoptic segmentation tasks. This model incorporates the Cellpose model for instance segmentation, followed by a classifier. Two classifiers currently available:

  - **FCNN (Fully Convolutional Neural Network)**: Operates directly on the patches or optionally on patches+masks.
  
  - **Random Forest**: Operates on the level of radiomic and intensity features extracted from the patches.

- **UNet**: Employed for semantic segmentation tasks. The UNet model is equipped with four layers and supports images of any size.

- **CellposeMultichannel**: Employed for panoptic segmentation task. The CellposeMultichannel model utilizes separate CustomCellposeModel instances for each channel and returns masks corresponding to each object type.


## Running with Docker [DO NOT USE UNTIL ISSUE IS SOLVED]

### Docker --> Currently doesn't work for generate labels? 

#### Docker-Compose
```
docker compose up
```
#### Docker Non-Interactively
```
docker build -t dcp-server .
docker run -p 7010:7010 -it dcp-server
```

#### Docker Interactively
```
docker build -t dcp-server .
docker run -it dcp-server bash
bentoml serve service:svc --reload --port=7010
```


