# Maritime Urban Tracking

This repository implements code to use the Maritime Urban Tracking (MUT) dataset. The dataset can be found at NIRD: [link](https://doi.org/10.11582/2025.l0rcnf5k), or the single maneuver 1 scenario [link](https://doi.org/10.11582/2026.dfc5yksv). The paper for the dataset will be found [below](#citations). 

Here is a gif of some of the data from the dataset, made using this library and a recording tool. 
The red dots are the LiDAR points and the othe colored points are from the short baseline stereo camera. 
The coordinate frames correspond to the LiDAR and camera frames. The x-axis is red, the y-axis is green and the z-axis is blue. 

![Ferry moving](./illustration/Point%20cloud%20when%20ferry%20moves%20reduced.gif)

Here is a gif where there is a moving target. The big green dots are the estimated GNSS locations. 
![Target moving](./illustration/Point%20cloud%20target%20moves%20reduced.gif)

## Pre-requisites
We assume that OpenCV is installed. Since we will allow for the CUDA-installation, we will not write opencv-python as a requirement. Copy your local cv2 installtion into the venv if you are using this. You might find it here: ``/usr/lib/python3.10/dist-packages/cv2`` and place it under ``venv/lib/python3.10/site-packages``. 

If you do not have OpenCV installed and you use python, you can install the CPU-version by ``pip install opencv-python==4.11.0.86``. 

We have tested this repository using Python 3.10. 

## Download the dataset
The dataset can be found here: [link](https://doi.org/10.11582/2025.l0rcnf5k). Note that you need to store both the compressed file and the resulting folder with all files, which is less than 1.6TB of data. 

We have also published a single one of the scenarios (Maneuver 1) here: [link](https://doi.org/10.11582/2026.dfc5yksv). This one is about 16GB. 

### Option 1: Use wget for the single scenario dataset (16GB)
1. Download: ``wget https://data.archive.sigma2.no/dataset/d3c38c16-e692-42d8-b785-68654a0d5439/download/Maritime%20Urban%20Tracking%20Maneuver%201.tar.gz``
1. Check hash: ``md5sum Maritime\ Urban\ Tracking\ Maneuver\ 1.tar.gz``. This should return ``355511bb15c57a20fc6b4fbc6c310ff0``
1. Extract: ``tar -xzvf Maritime\ Urban\ Tracking\ Maneuver\ 1.tar.gz``

### Option 2: Use wget for the full dataset (800GB)
1. Download: ``wget https://data.archive.sigma2.no/dataset/6723ebbe-2505-4321-b94d-8a4e482cd6bb/download/Maritime-Urban-Tracking.tar.gz``
1. Check hash: ``md5sum Maritime-Urban-Tracking.tar.gz``. This should return ``90546e78aeee116ad3630bb95e442a5d``
1. Extract: ``tar -xzvf Maritime-Urban-Tracking.tar.gz``

### Option 3: Download via web browser
1. Go to the dataset website linked above.
1. Click on ``Data Access/Table Of Contents`` and click on the ``.tar.gz``-file to download it. 
1. After downloading, extract the contents, e.g., run ``tar -xzvf Maritime-Urban-Tracking.tar.gz``. 

## Installation
1. Download the repository. 
1. Consider using a venv like ``python3 -m venv venv`` and source it, ``source venv/bin/activate``. 
1. Install our code using
    ```bash
    pip install -e .
    ```
1. Ensure that OpenCV is installed, see pre-requisite above. 
1. Then change the `DATASET_FOLDER` variable in [this file](src/maritime_urban_tracking/sequences.py#L6) to the correct absolute path. 

## Usage
Once the the installation is done, the code can be used as a regular python library. You may run the examples, like [this one](examples/play_all_together.py), or use it in your own code similarly to how it is done in the examples: ``from maritime_urban_tracking.lidar import Lidar``. 

Example:
```
python3 examples/play_all_together.py
```
or run the debugger in vscode. 

If you want to use tha dataset, but not in python, you may use the code in this repository as a specification of how to read the files. It would be nice if you then also make that interface available to others and make an issue in this repository so that we can link to it. 

## More documentation
To read more details about the dataset:
1. (The most detailed option) read the code and the data files
1. Read the paper, see below. 
1. Read more details in [this README.md](documentation/README.md)

## Citations
If you find the dataset useful in your research, please cite our paper. We will add a suggestion of how to cite the paper here once it is published. 
