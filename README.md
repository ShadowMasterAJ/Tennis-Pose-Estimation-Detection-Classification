# TennisPose: A Multi-Task Deep Learning Approach for Player Action Detection, Pose Estimation, and Classification in Tennis

## Overview
This repository contains the codebase for experiments with different model architectures to do pose estimation, player detection and shot classification in tennis. Architectures include models such as YOLO11m-Pose, EfficientNet Backbone and techniques such as Squeeze-and-Excitation, Multi-scale Fusion, LSTM and SpatioTemporal Attenion too.

## Project Structure
- `checkpoints_conv/`: Contains checkpoints for convolutional models.
- `checkpoints_net/`: Contains checkpoints for network models.
- `checkpointsEnhanced/`: Contains enhanced checkpoints.
- `checkpointsRes/`: Contains residual checkpoints.
- `dataset_yolo/`: Contains YOLO dataset files.
    - `dataset.yaml`: Configuration file for the YOLO dataset.
    - `images/`: Directory for dataset images.
    - `labels/`: Directory for dataset labels.
- `logs/`: Contains logs for training and evaluation.
    - `logs_simple/`: Logs for simple tennis conv.
    - `logs_enhanced/`: Logs for enhanced tennis conv.
    - `logs_net/`: Logs for tennis net.
- `models/`: Contains model implementations.
- `og_dataset/`: Contains the original dataset.
    - `annotations/`: Directory for dataset annotations.
    - `images/`: Directory for dataset images.
- `tennis_pose_estimation/`: Contains directories for different pose estimation configurations.
- `training/`: Contains training scripts and configurations.
- `utils/`: Contains utility scripts.

## Installation
To run the code, you need to have Python 3.x installed. You can install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
Note that the log files are tensorboard event files and can be visualised using 

```bash
tensorboard --logdir <dir name>
```

## Contributors
- Arnav [GitHub Profile](https://github.com/ShadowMasterAJ)
