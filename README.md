# Dota Learning to Understand Aerial Image 2021 Challenge using Detectron2

> Goal: train a detector to detect the orientation of rotated objects in DOTAv2 Aerial Image 2021 Challenge

The project is for the __2021 Learning to Understand Aerial Images Challenge on DOTA dataset__ focused on training and benchmarking challenges for object detection in aerial images. 

This project development utilizes the [detectron2](https://github.com/facebookresearch/detectron2) as the main modeling framework. Additionally, all models utilize detectron2 baseline [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) models that contains the configs and models for transfer learning.

## Project Overview and Challenge Result on Task 1

Training with DOTA-v2.0 train and validation datasets, we utilized detectron2 equipped with a ResNeXt-32x8d-101 backbone to detect the orientation of rotated objects in the Learning to Understand Aerial Image 2021 Task 1 challenge. The two-stage detection approach for ResNeXt was selected primarily due to the ability to process higher quality features with better bounding box refinement. Implementing the Rotation Region Proposal Network (RRPN) assisted the network to learn and detect the orientation of arbitrary aerial objects that make use of bounding box regression to more accurate region of interest object detection. Additionally, we also set the threshold for IoU in the RRPN to 0.7 to incorporate negative samples in the training process. 

Due to training and evaluation challenge timeline, experiment with multi-scaling and augmentation was primarily focused where imbalanced in the dataset existed. However, data augmentation was for the training dataset was used minimal (some rotate and lighting). The training and test-challenge dataset were rescaled and cropped to patches of 1024 x 1024 based off the original image with 200 crop strides. 


### Evaluation for Task 1


| Approaches | mAP | PL | BD | BR | GTF | SV | LV | SH | TC | BC | ST | SBF | RA | HA | SP | HC | CC | AIR | HP |
|------------|:---:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|Rotated ResNeXt 101|0.19|0.52|0.29|0.13|0.46|0.17|0.09|0.22|0.15|0.09|0.35|0.16|0.35|0.11|0.01|0.00|0.00|0.23|0.02|

### Summary Detections Results

![](docs/media/20k_dotav2_1024split_val_080421_dotatrainval_devkit_frcnn_X101_nc24x01.png)


## Getting Started

* [Installation_doc](docs/01_Installation.md) instructions
* [Dataset](docs/02_Dataset_Details.md) detail procedures for downloading and preparing the dataset
* To start training [Quickstart.md](docs/03_Quickstart.md)
* [Acknowledgements.md](docs/Acknowledgements.md)

