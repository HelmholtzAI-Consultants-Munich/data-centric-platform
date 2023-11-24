# DCP Client
A data centric platform for microscopy imaging

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/HelmholtzAI-Consultants-Munich/active-learning-platform)
![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)

## How to use?
### Installation
This installation has been tested using a conda environment with python version 3.9 on a mac local machine. In your dedicated environment run:
```
pip install -e .
```

### Running the client: A step to step guide!
1. Before launching the GUI you will need to set up your client configuration file, _dcp_client/config.cfg_. Please, obey the [formal JSON format](https://www.json.org/json-en.html). Here, we will define how the client will interact with the server. There are currently two options available: running the server locally, or connecting to the running instance on the FZJ jusuf-cloud. To connect to a locally running server, set:
  ```
     "user": "local", 
      "host": "local", 
      "data-path": "None", 
      "ip": "localhost", 
      "port": 7010
  }
  ```
  To connect to the running service on jusuf-cloud, set:
  ```
       "server":{
        "user": "xxxxx",
        "host": "xxxxxx", 
        "data-path": "xxxxx",
        "ip": "xxx.xx.xx.xx", 
        "port": xxxx
    }
  ```
  Before continuing, you need to make sure that DCP server is running, either locally or on the cloud. See [DCP Server Installation & Launch](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md#using-pypi) for instructions on how to launch the server. **Note:** In order for this connection to succeed, you will need to have contacted the team developing DCP, so they can add your IP to the list of accepted requests.


2. After setting your config simply run:
  ```
  python dcp_client/main.py
  ```

3. The welcome window should have now popped up.
   
   <img src="https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/documentation/src/client/readme_figs/client_welcome_window.png"  width="400" height="200">
  Here you will need to select the directories which we will be using throughout the data centric workflow. The following directories need to be defined:
  * **Uncurated dataset path:** This folder should initially contain all images of your dataset. They may or may not be accompanied by corresponding segmentations, but if they do, the segmentations should have the same filename as the image followed by the ending defined in ```setup/seg_name_string```, deifned in ```server/dcp_server/config.cfg``` (default extension is _seg)
  * **Curation in progress path:(Optional)** Images for which the segmentation is a work in progress should be moved here. Each image in this folder can have one or multiple segmentations corresponding to it (by changing the filename of the segmentation in the napari layer list after editing it, see ...). If you do not want to use an intermediate working dir, you can skip setting a path to this directory (it is not required).
  * **Curated dataset path:** This folder should contain images along with their final segmentations. **Only** move images here when the segmentation is complete and finalised, you won't be able to change them after they have been moved here. These are then used for training your model.

4. After setting the paths for these three folders, you can click the **Start** button. If you have set the server configuration to the cloud, you will receive a message notifying you that your data will be uploaded to the cloud. Clik **Ok** to continue.

5. The main working window will appear next. This gives you an overview of the directories selected in the previous step along with three options:
   ![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/documentation/src/client/readme_figs/client_data_overview_window.png)
   * **Generate Labels:** Click this button to generate labels for all images in the "Uncurated dataset" directory. This will call the ```segment_image``` service from the server.
   * **View image and fix label:** Click this button to launch your viewer. The napari software is used for visualising, and editing the images segmentations. See ...
   * **Train Model:** Click this model to train your model on the images in the "Curated dataset" directory. This will call the ```train``` service from the server.
6. The viewer.
7. Data centric workflow [intended usage]:
   
