# mti-anomaly-detection

Anomaly detection using [DeepLabv3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) with fine tuning. Adapted from https://github.com/msminhas93/DeepLabv3FineTuning.

## Installation

- Download the [MVTec AD](https://www.kaggle.com/datasets/ipythonx/mvtec-ad) dataset on Kaggle.
- Copy and paste the wood subfolder in `data/raw/wood`.
- Create a `data/raw/holes/Images` and `data/raw/holes/Masks` by pasting images respectively from `data/raw/wood/test/hole` and `data/raw/wood/ground_truth/hole`.

## Usage

The Docker file helps create an image to run a container on GCP. This whole process can be replaced using AutoML on VertexAI. The `main.py` file should be run on MPS device while the `main_gcp.py` file should be run on CUDA device (GCP TPU).