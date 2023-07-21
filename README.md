[![DeepD3 project website](https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)](https://deepd3.forschung.fau.de/)
[![Documentation Status](https://readthedocs.org/projects/deepd3/badge/?version=latest)](https://deepd3.readthedocs.io/en/latest/?badge=latest)

# 3D ROI Generation of dendritic spines using DeepD3

This repository provides an implementation of a Dendritic Spine Detection system using the Deep3D framework. The system is capable of detecting dendritic spines and generating 3D Regions of Interest (ROIs) of these spines for further analysis.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

1. Python 3.7 or later.
2. Other dependencies specified in the requirements.txt file.

# DeepD3

DeepD3, a dedicated framework for the segmentation and detection of dendritic spines and dendrites.

Utilizing the power of deep learning, DeepD3 offers a robust and reliable solution for neuroscientists and researchers interested in analyzing the intricate structures of neurons.

With DeepD3, you are able to

* train a deep neural network for dendritic spine and dendrite segmentation
* use pre-trained DeepD3 networks for inference
* build 2D and 3D ROIs
* export results to your favourite biomedical image analysis platform
* use command-line or graphical user interfaces

## How to install and run DeepD3

DeepD3 is written in Python. First, please download and install any Python-containing distribution, such as [Anaconda](https://www.anaconda.com/products/distribution). We recommend Python 3.7 and more recent version.

Then, installing DeepD3 is as easy as follows:

    pip install deepd3

Now, you have access to almost all the DeepD3 functionalities.

If you want to use the DeepD3 Neural Network inference mode, please install **Tensorflow** using either of the following commands:

    # With CPU support only
    conda install tensorflow

    # With additional GPU support
    conda install tensorflow-gpu

If you would like to access DeepD3-GUIs, use the following two shortcuts in your favourite shell:

    # Opening the segmentation and ROI building GUI
    deepd3-inference

    # Opening the training utilities
    deepd3-training

## Model zoo

We provide a comprehensive training dataset on zenodo and the [DeepD3 Website](https://deepd3.forschung.fau.de/):

* [DeepD3_8F.h5](https://deepd3.forschung.fau.de/models/DeepD3_8F.h5) - 8 base filters, original resolution
* [DeepD3_16F.h5](https://deepd3.forschung.fau.de/models/DeepD3_16F.h5) - 16 base filters, original resolution
* [DeepD3_32F.h5](https://deepd3.forschung.fau.de/models/DeepD3_32F.h5) - 32 base filters, original resolution
* [DeepD3_8F_94nm.h5](https://deepd3.forschung.fau.de/models/DeepD3_8F_94nm.h5) - 8 base filters, resized to 94 nm xy resolution
* [DeepD3_16F_94nm.h5](https://deepd3.forschung.fau.de/models/DeepD3_16F_94nm.h5) - 16 base filters, resized to 94 nm xy resolution
* [DeepD3_32F_94nm.h5](https://deepd3.forschung.fau.de/models/DeepD3_32F_94nm.h5) - 32 base filters, resized to 94 nm xy resolution

Brief description:

* Full (32F) DeepD3 model trained on 94 nm (fixed) or a blend of resolutions (free)
* Medium (16F) DeepD3 model trained on 94 nm (fixed) or a blend of resolutions (free)
* Tiny (8F) DeepD3 mode trained on 94 nm (fixed) or a blend of resolutions (free)


## Workflow

### Training and Validation Dataset Guide

This guide will show you how to access and use the DeepD3 training and validation datasets from the DeepD3 website for dendritic spine and dendrite detection.

Downloading Datasets

* Visit the [DeepD3 Website](https://deepd3.forschung.fau.de/)
* Navigate to the `Datasets` section.
* Look for the `DeepD3_Training.d3set` and `DeepD3_Validation.d3set` datasets. They should be clearly labeled.
* Click on the download link or button for each dataset.
* Save the datasets in your local directory or in a directory accessible to your Python environment.

### Train DeepD3 on your own dataset

We have prepared a Jupyter notebook `Training_deepd3.ipynb` in the folder `examples`. Follow the instructions to train your own deep neural network for DeepD3 use.
Steps we follow below:
* Import all necessary libraries. This includes TensorFlow and other related packages.
* Specify the paths for your training and validation datasets and prepare your datasets.

``` from deepd3.training.stream import DataGeneratorStream

# Specify the paths to your training and validation datasets
TRAINING_DATA_PATH = r"/path/to/your/downloaded/training/data"
VALIDATION_DATA_PATH = r"/path/to/your/downloaded/validation/data"

# Create data generators
dg_training = DataGeneratorStream(TRAINING_DATA_PATH, batch_size=32, target_resolution=0.094, min_content=50)
dg_validation = DataGeneratorStream(VALIDATION_DATA_PATH, batch_size=32, target_resolution=0.094, min_content=50, augment=False, shuffle=False)
```
Replace `/path/to/your/downloaded/training/data` and `/path/to/your/downloaded/validation/data` with the actual paths to your downloaded training and validation datasets.

* Visualize your input data to ensure it has been loaded correctly.
* Initialize your DeepD3 model with appropriate settings.
* Specify the callbacks and train your model.
* Plot the loss and accuracy metrics for both training and validation sets to evaluate your model.

This guide should get you started with training a deep learning model using DeepD3. If you have any questions, feel free to open an issue on this repository.

### Inference

Open the inference mode using `deepd3-inference`. Load your stack of choice (we currently support TIF stacks) and specify the XY and Z dimensions. Next, you can segment dendrites and dendritic spines using a DeepD3 model from [the model zoo]() by clicking on `Analyze -> Segment dendrite and spines`. Afterwards, you may clean the predictions by clicking on `Analyze -> Cleaning`. Finally, you may build 2D or 3D ROIs using the respective functions in `Analyze`. To test the 3D ROI building, double click in the stack to a region of interest. A window opens that allows you to play with the hyperparameters and segments 3D ROIs in real-time.

All results can be exported to various file formats. For convenience, DeepD3 saves related data in its "proprietary" hdf5 file (that you can open using any hdf5 viewer/program/library). In particular, you may export the predictions as TIF files, the ROIs to ImageJ file format or a folder, the ROI map to a TIF file, or the ROI centroids to a file. 

Most functions can be assessed using a batch command script located in `deepd3/inference/batch.py`.


## How to cite

DeepD3 is available as preprint soon.
