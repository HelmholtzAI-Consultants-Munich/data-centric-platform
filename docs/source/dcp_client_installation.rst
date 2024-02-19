.. _DCP Client:

DCP Client
===========

The client of our data centric platform for microscopy imaging.

.. image:: https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg
   :alt: stability-wip

Installation
-------------

Before starting make sure you have navigated to ``data-centric-platform/src/client``. All future steps expect you are in the client directory. This installation has been tested using a conda environment with python version 3.9 on a mac local machine. In your dedicated environment run:

.. code-block:: bash

   pip install -e .

Running the client: A step-by-step guide!
------------------------------------------

1. **Configurations**
~~~~~~~~~~~~~~~~~~~~~~~~

   Before launching the GUI you will need to set up your client configuration file, _dcp_client/config.cfg_. Please, obey the `formal JSON format <https://www.json.org/json-en.html>`_. Here, we will define how the client will interact with the server. There are currently two options available: running the server locally, or connecting to the running instance on the FZJ jusuf-cloud. To connect to a locally running server, set:

   .. code-block:: json

      "server":{
                 "user": "local",
                 "host": "local",
                 "data-path": "None",
                 "ip": "localhost",
                 "port": 7010
      }

   To connect to the running service on jusuf-cloud, set:

   .. code-block:: json

        "server":{
               "user": "xxxxx",
               "host": "xxxxxx",
               "data-path": "xxxxx",
               "ip": "xxx.xx.xx.xx",
               "port": xxxx
        }

   Before continuing, you need to make sure that DCP server is running, either locally or on the cloud. See `DCP Server Installation & Launch <https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md#using-pypi>`_ for instructions on how to launch the server. **Note:** In order for this connection to succeed, you will need to have contacted the team developing DCP, so they can add your IP to the list of accepted requests.

   To make it easier for you we provide you with two config files, one works when running a local server and one for remote - just make sure you rename the config file you wish to use to ``config.cfg``. The default is local configuration.

2. **Launching the client**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   After setting your config simply run:

   .. code-block:: bash

      python dcp_client/main.py

3. **Welcome window**
~~~~~~~~~~~~~~~~~~~~~~

   The welcome window should have now popped up.
 
   .. image:: https://raw.githubusercontent.com/HelmholtzAI-Consultants-Munich/data-centric-platform/main/src/client/readme_figs/client_welcome_window.png
         :width: 400
         :height: 200
         :align: center


   Here you will need to select the directories which we will be using throughout the data centric workflow. The following directories need to be defined:

   - **Uncurated Dataset Path:**
   
   This folder is intended to store all images of your dataset. These images may be accompanied by corresponding segmentations. If present, segmentation files should share the same filename as their associated image, appended with a suffix as specified in ``server/dcp_server/config.cfg`` file (default: '_seg').

   - **Curation in Progress Path (Optional):**

   Images for which the segmentation is a work in progress should be moved here. Each image in this folder can have one or multiple segmentations corresponding to it (by changing the filename of the segmentation in the napari layer list after editing it, see **Viewer**). If you do not want to use an intermediate working dir, you can skip setting a path to this directory (it is not required). No future functions affect this directory, it is only used to move to and from the uncurated and curated directories.

   - **Curated Dataset Path:**

   This folder is intended to contain images along with their final segmentations. **Only** move images here when the segmentation is complete and finalised, you won't be able to change them after they have been moved here. These are then used for training your model.

4. **Setting paths**
~~~~~~~~~~~~~~~~~~~~~

   After setting the paths for these three folders, you can click the **Start** button. If you have set the server configuration to the cloud, you will receive a message notifying you that your data will be uploaded to the cloud. Click **Ok** to continue.

5. **Data Overview**
~~~~~~~~~~~~~~~~~~~~

   The main working window will appear next. This gives you an overview of the directories selected in the previous step along with three options:

   - **Generate Labels:** Click this button to generate labels for all images in the "Uncurated dataset" directory. This will call the ``segment_image`` service from the server
   - **View image and fix label:** Click this
