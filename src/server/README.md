# DCP Server

The client and server communicate via the [bentoml](https://www.bentoml.com/?gclid=Cj0KCQiApKagBhC1ARIsAFc7Mc6iqOLi2OcLtqMbGx1KrFjtLUEZ-bhnqlT2zWREE0x7JImhtNmKlFEaAvSSEALw_wcB) library. The client interacts with the server every time we run model inference or training, so the server should be running before starting the client.

## How to use?

### Installation
In your dedicated environment run:
```
pip install -e .
```

#### Launch DCP server
Simply run:
```
python dcp_server/main.py
```
Once the server is running, you can verify it is working by visiting http://localhost:7010/ in your web browser.

## Customization (for developers)

All service configurations are set in the _config.cfg_ file. Please, obey the [formal JSON format](https://www.json.org/json-en.html).

The config file has to have the five main parts. All the ```marked``` arguments are mandatory:

 - ``` setup ``` 
    - ```segmentation ``` - segmentation type from the segmentationclasses.py. Currently, only **GeneralSegmentation** is available (MitoProjectSegmentation and GFPProjectSegmentation are stale). 
    - ```accepted_types``` - types of images currently accepted for the analysis
    - ```seg_name_string``` - end string for masks to run on (All the segmentations of the image should contain this string - used to save and search for segmentations of the images)
- ```service```
    - ```model_to_use``` - name of the model class from the models.py you want to use. Currently, available models are:
      -  **CustomCellposeModel**: Inherits [CellposeModel](https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel) class
      -  **CellposePatchCNN**: Includes a segmentor and a clasifier. Currently segmentor can only be ```CustomCellposeModel```, and classifier is ```CellClassifierFCNN```. The model sequentially runs the segmentor and then classifier, on patches of the objects to classify them.
    - ```save_model_path``` - name for the trained model which will be saved after calling the (re)train from service - is saved under ```bentoml/models```
    - ```runner_name``` -  name of the runner for the bentoml service 
    - ```service_name``` - name for the bentoml service
    - ```port``` - on which port to start the service
- ```model``` - configuration for the model instatiation. Here, pass any arguments you need or want to change. Take care that the names of the arguments are the same as of original model class' _init()_ function!
  - ```segmentor```: model configuration for the segmentor. Currently takes argumnets used in the init of CellposeModel, see [here](https://cellpose.readthedocs.io/en/latest/api.html#cellposemodel).
  - ```classifier```: model configuration for classifier, see _init()_ of ```CellClassifierFCNN``` 
- ```train``` - configuration for the model training. Take care that the names of the arguments are the same as of original model's _train()_ function!
  - ```segmentor```: If using cellpose - the _train()_ function arguments can be found [here](https://cellpose.readthedocs.io/en/latest/api.html#id7). Here, pass any arguments you need or want to change or leave empty {}, then default arguments will be used.
  - ```classifier```: train configuration for classifier, see _train()_ of ```CellClassifierFCNN``` 
- ```eval``` - configuration for the model evaluation.. Take care that the names of the arguments are the same as of original model's _eval()_ function! 
  - ```segmentor```: If using cellpose - the _eval()_ function arguments can be found [here](https://cellpose.readthedocs.io/en/latest/api.html#id3). Here, pass any arguments you need or want to change or leave empty {}, then default arguments will be used.
  - ```classifier```: train configuration for classifier, see _eval()_ of ```CellClassifierFCNN```.
  - ```mask_channel_axis```: If a multi-class instance segmentation model has been used, then the masks returned by the model should have two channels, one for the instance segmentation results and one indicating the obects class. This variable indicated at which dim the channel axis should be stored. Currently should be kept at 0, as this is the only way the masks can be visualised correcly by napari in the client.


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


