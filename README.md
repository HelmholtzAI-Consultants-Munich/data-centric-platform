# Data Centric Platform
*A data-centric platform for all-kinds segmentation in microscopy imaging.*

![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
![tests](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/actions/workflows/test.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/HelmholtzAI-Consultants-Munich/data-centric-platform)
[![Documentation Status](https://readthedocs.org/projects/data-centric-platform/badge/?version=latest)](https://data-centric-platform.readthedocs.io/en/latest/?badge=latest)

## Overview

This repository includes both a **server** and **client** side implementation of our Data Centric Platform (DCP) for microscopy imaging. The platform supports:
- **Instance segmentation**
- **Semantic segmentation**
- **Multi-class instance segmentation**

The client and server communicate via the [BentoML](https://www.bentoml.com/) library. For full functionality, the server must be running (locally or remotely) before launching the client.

<img src="https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/readme_figs/dcp_pipeline.png"  width="200" height="200">

---

## 🚀 Getting Started

To get up and running with the Data Centric Platform:

### 1. [Install & Launch the Server](#dcp-server)
Make sure you start the server before running the client. You can find full instructions below or in the [DCP Server README](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md#using-pypi).

### 2. [Install & Launch the Client](#dcp-client)
Once the server is running, install and start the client using the GUI. More details are below or in the [DCP Client README](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/README.md).

📝 **Need a step-by-step visual guide?**  
Check out our [official documentation website](https://data-centric-platform.readthedocs.io/en/latest/dcp_client_installation.html) for a detailed walkthrough of how to install, configure, and run the client.

---

## 📁 Toy Data

We include a `data/` folder in this repo with toy data for testing. You can use this as the *Uncurated dataset* in the welcome window and create empty folders for the other two required directories to start experimenting.

---

## 🧠 Data-Centric Development

Our platform supports best practices in data-centric AI:
- **Detect and remove outliers** from your training data: only confirmed samples are used to train our models
- **Detect and correct labeling errors**: editing labels with the integrated napari visualisation tool
- **Establish consensus**: allows for multiple annotators before curated label is passed to train model
- **Focus on data curation**: no interaction with model parameters during training and inference

---

## 🖥 DCP Server

The server handles model training and inference tasks.

### Installation (Tested on Python 3.9–3.11)
```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform.git
cd data-centric-platform/src/server
pip install numpy
pip install -e .
```

### Launch the Server
```bash
python dcp_server/main.py
```
Visit [http://localhost:7010/](http://localhost:7010/) to verify the server is running.

📄 Full instructions available in the [Server README](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/server/README.md).

📚 [Server Documentation](https://data-centric-platform.readthedocs.io/en/latest/dcp_server_installation.html)

---

## 🧑‍💻 DCP Client

The client provides a user-friendly GUI to interact with the server for data curation, labeling, training, and inference.

### Installation (Tested on Python 3.9–3.12)
```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform.git
cd data-centric-platform/src/client
pip install -e .
```

### Launch the Client
Make sure the server is already running, then start the client with:

```bash
dcp-client --mode local
```
or

```bash
dcp-client --mode remote
```

📄 Full instructions in the [Client README](https://github.com/HelmholtzAI-Consultants-Munich/data-centric-platform/blob/main/src/client/README.md)

📚 [Client Documentation](https://data-centric-platform.readthedocs.io/en/latest/dcp_client_installation.html)

---

## 📚 Documentation

Explore the full platform features and usage examples:  
📖 [Documentation Homepage](https://data-centric-platform.readthedocs.io/en/latest/index.html)

---

## ✨ Get More With Less!

Let your data do the talking. Let DCP help you curate, clean, and improve your datasets—so your models can shine.