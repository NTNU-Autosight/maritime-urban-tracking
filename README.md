# Maritime Urban Tracking

**IMPORTANT NOTE**: This is an archived version of the code. An up-to-date version will be placed on GitHub: [link](https://github.com/NTNU-Autosight/maritime-urban-tracking). 

This repository implements code to use the Maritime Urban Tracking (MUT) dataset. The dataset can be found at NIRD: [link](https://archive.sigma2.no/). A specific link will be added when the dataset is published in the up-to-date repository. The paper for the dataset will be found [below](#citations). 

Here is a gif of some of the data from the dataset, made using this library and a recording tool. 
The red dots are the LiDAR points and the othe colored points are from the short baseline stereo camera. 
The coordinate frames correspond to the LiDAR and camera frames. 
![Ferry moving](./illustration/Point%20cloud%20when%20ferry%20moves%20reduced.gif)

Here is a gif where there is a moving target. The big green dots are the estimated GNSS locations. 
![Target moving](./illustration/Point%20cloud%20target%20moves%20reduced.gif)

## Pre-requisites
We assume that OpenCV is installed. Since we will allow for the CUDA-installation, we will not write opencv-python as a requirement. Copy your local cv2 installtion into the venv if you are using this. You might find it here: ``/usr/lib/python3.10/dist-packages/cv2`` and place it under ``venv/lib/python3.10/site-packages``. 

If you do not have OpenCV installed and you use python, you can install the CPU-version by ``pip install opencv-python``. 

We have tested this repository using Python 3.10. 

## Installation
1. Download the repository. 
1. Consider using a venv like ``python3 -m venv venv`` and source it, ``source venv/bin/activate``. 
1. Install our code using
    ```bash
    pip install -e .
    ```
1. Ensure that OpenCV is installed, see pre-requisite above. 
1. Then change to the correct dataset location in [this file](src/maritime_urban_tracking/sequences.py#L6)

## Usage

Once the the installation is done, the code can be used as a regular python library. You may run the examples, like [this one](examples/play_all_together.py), or use it in your own code similarly to how it is done in the examples: ``from maritime_urban_tracking.lidar import Lidar``. 

If you want to use tha dataset, but not in python, you may use the code in this repository as a specification of how to read the files. It would be nice if you then also make that interface available to others and make an issue in this repository so that we can link to it. 

## Citations

If you find the dataset useful in your research, please cite our paper. A suggestion of how to cite it will be given in the up-to-date repository. 
