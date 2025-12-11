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

* **Instance** segmentation:

  Make sure the server is already running, either locally or remotely. Then, depending on the configuration, simply run:
  ```
  dcp-client --mode local
  ```
  or 
  ```
  dcp-client --mode remote
  ```
  depending on whether your server is running locally or remotely.

* **Multi-class instance** segmentation:

  When you additionally have data with multiple classes you will need to define this on runtime. You will need to add the ```--multi-class``` and ```--num-classes``` arguments to you run and specify the number of classes in your data. For example, for a multi-class problem with three classes, run:
   ```
  dcp-client --mode local --multi-class --num-classes 3
  ```

### SAM Assisted Labelling

The client includes **SAM (Segment Anything Model)** integration for AI-assisted segmentation. This feature helps you quickly annotate objects by drawing bounding boxes or clicking points.

#### Enabling SAM

1. Open an image in the Napari viewer
2. Toggle the **"Assisted labelling"** button to **ON**
3. The SAM model will load (first use may take a moment to download the model checkpoint)

#### Prompting Modes

**Box Prompting (default):**
- Select "Boxes" in the prompting options
- Draw bounding boxes around objects you want to segment
- SAM will automatically generate a segmentation mask for each box
- Press **Enter** to accept all masks, or **Escape** to reject the last mask
- Press **Ctrl+Z** to undo the last box

**Point Prompting:**
- Select "Points" in the prompting options
- Click inside objects to add **foreground points** (white)
- Press **'b'** to toggle to **background point mode** (red) for refining masks
- Press **'d'** to confirm points and generate the mask
- Press **Enter** to accept, **Escape** to reject

#### Important Notes

- SAM is only available on the **Instance channel** (channel 0) in multi-class mode
- When switching to the Labels channel, SAM is automatically disabled
- SAM state is restored when switching back to the Instance channel
- Hover over the prompting options in the UI for additional usage tips

## Want to know more?
Visit our [documentation](https://data-centric-platform.readthedocs.io/en/latest/dcp_client_installation.html) for more information and a step by step guide on how to run the client.
