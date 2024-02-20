# DCP Client
The client of our data centric platform for microscopy imaging.

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)

## How to use?
### Installation
Before starting, make sure you have navigated to ```data-centric-platform/src/client```. All future steps expect you are in the client directory. This installation has been tested using a conda environment with python version 3.9 on a mac local machine. In your dedicated environment run:
```
pip install -e .
```

### Running the client: A step to step guide!
1. **Configurations**
   
Before launching the GUI you will need to set up your client configuration file, _dcp_client/config.cfg_. Please, obey the [formal JSON format](https://www.json.org/json-en.html). Here, we will define how the client will interact with the server. There are currently two options available: running the server locally, or connecting to the running instance on the FZJ jusuf-cloud. To connect to a locally running server, set:
  ```
     "server":{
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
              "user": "rocky",
              "host": "jsc-vm", 
              "data-path": "/home/rocky/dcp-data/my-project",
              "ip": "134.94.198.230", 
              "port": 7010
    }
  ```
  Before continuing, you need to make sure that DCP server is running, either locally or on the cloud. See [DCP Server Installation & Launch](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md#using-pypi) for instructions on how to launch the server. **Note:** In order for this connection to succeed, you will need to have contacted the team developing DCP, so they can add your IP to the list of accepted requests.

To make it easier for you we provide you with two config files, one works when running a local server and one for remote - just make sure you rename the config file you wish to use to ```config.cfg```. The defualt is local configuration. 


2. **Launching the client**
   
After setting your config simply run:
  ```
  python dcp_client/main.py
  ```

3. **Welcome window**
   
The welcome window should have now popped up.
   
   <img src="https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/readme_figs/client_welcome_window.png"  width="400" height="200">
   
  Here you will need to select the directories which we will be using throughout the data centric workflow. The following directories need to be defined:
  
  * **Uncurated dataset path:** This folder is intended to store all images of your dataset. These images may be accompanied by corresponding segmentations. If present, segmentation files should share the same filename as their associated image, appended with a suffix as specified in 'setup/seg_name_string', defined in ```server/dcp_server/config.cfg``` (default: '_seg').
  * **Curation in progress path:(Optional)** Images for which the segmentation is a work in progress should be moved here. Each image in this folder can have one or multiple segmentations corresponding to it (by changing the filename of the segmentation in the napari layer list after editing it, see **Viewer**). If you do not want to use an intermediate working dir, you can skip setting a path to this directory (it is not required). No future functions affect this directory, it is only used to move to and from the uncurated and curated directories.
  * **Curated dataset path:** This folder is intended to contain images along with their final segmentations. **Only** move images here when the segmentation is complete and finalised, you won't be able to change them after they have been moved here. These are then used for training your model.

4. **Setting paths**

After setting the paths for these three folders, you can click the **Start** button. If you have set the server configuration to the cloud, you will receive a message notifying you that your data will be uploaded to the cloud. Clik **Ok** to continue.

5. **Data Overview**
   
The main working window will appear next. This gives you an overview of the directories selected in the previous step along with three options:

   * **Generate Labels:** Click this button to generate labels for all images in the "Uncurated dataset" directory. This will call the ```segment_image``` service from the server
   * **View image and fix label:** Click this button to launch your viewer. The napari software is used for visualising, and editing the images segmentations. See **Viewer**
   * **Train Model:** Click this model to train your model on the images in the "Curated dataset" directory. This will call the ```train``` service from the server
   ![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/readme_figs/client_data_overview_window.png)
   
6. **The viewer**

In DCP, we use [napari](https://napari.org/stable) for viewing our images and makss, adding, editing or removing labels. An example of the viewer can be seen below. After adding or removing any objects and editing existing objects wherever necessary, there are two options available:
- Click the **Move to Curation in progress folder** if you are not 100% certain about the labels you have created. You can also click on the label in the labels layer and change the name. This will result in several label files being created in the *In progress folder*, which can be examined later on.  **Note:** When changing the layer name in Napari, the user should rename it such that they add their initials or any other new info after _seg. E.g., if the labels of 1_seg.tiff have been changed in the Napari viewer, then the appropriate naming would for example be: 1_seg_CB.tiff and not 1_CB_seg.tiff.
- Click the **Move to Curated dataset folder** if you are certain that the labels you are now viewing are final and require no more curation. These images and labels will later be used for training the machine learning model, so make sure that you select this option only if you are certain about the labels. If several labels are displayed (opened from the 'Curation in progress' step), make sure to **click** on the single label in the labels layer list you wish to be moved to the *Curated data folder*. The other images will then be automatically deleted from this folder.

![Alt Text](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/readme_figs/client_napari_viewer.png)

### Data centric workflow [intended usage summary]
The intended usage of DCP would include the following:
1. Setting up configuration, run client (with server already running) and select data directories
2. Generate labels for data in *Uncurated data folder*
3. Visualise the resulting labels with the viewer and correct labels wherever necessary - once done move the image *Curated data folder*. Repeat this step for a couple of images until a few are placed into the *Curated data folder*. Depending on the qualitative evaluation of the label generation you might want to include fewer or more images, i.e. if the resulting masks require few edits, then few images will most likely be sufficient, whereas if many edits to the mask are required it is likely that more images are needed in the *Curated data folder*. You can always start with a small number and adjust later
4. Train the model with the images in the *Curated data folder*
6. Repeat steps 2-4 until you are satisfied with the masks generated for the remaining images in the *Uncurated data folder*. Every time the model is trained in step 4, the masks generated in step 2 should be of higher quality, until the model need not be trained any more 
<img src="https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/readme_figs/dcp_pipeline.png"  width="200" height="200">

   
