# Dota Leanring to Understand Aerial Image 2021 Challenge using Detectron2

> Goal: train a detector to detect the orientation of rotated objects in DOTAv2 Aerial Image 2021 Challenge

The project is for the __2021 Learning to Understand Aerial Images Challenge on DOTA dataset__ focused on training and benchmarking challenges for object detection in aerial images. 

This project development utilizes the [detectron2](https://github.com/facebookresearch/detectron2) as the main modeling framework. Additionally, all models utilize detectron2 baseline [model zoo](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md) models that contains the configs and models for transfer learning.

## Main Results

| Approaches                    | mAP | Plane | Ship | tank | harbor | helipad |
|------------                   |:---:|:-----:|:----:|:--:|:---:|:--:|
|y5_dota2_split1024_071621_dev  |TBD |    0.09| 0.57 | 0.18 | 0.0 | 0.0 |



### Summary Detections Results

![](docs/media/20k_dotav2_1024split_val_080421_dotatrainval_devkit_frcnn_X101_nc24x01.png)


## Getting Started

* [Installation_doc](docs/01_Installation.md) instructions
* [Dataset](docs/02_Dataset_Details.md) detail procedures for downloading and preparing the dataset

## Dataset toolkit

In summary the implementation of the dataset pre-processing before training include the following steps:

1. Split and Crop Images
2. Reformat the data from the DOTA txt to COCO json
3. visualize the new dataset and validate the rotated box format is accurate
4. update any config needed for processing the new input dataset. The base defaults set in this repo should work fine without any modification, except for updating the required dataset path directories. 

### Prepare DOTA dataset

Here is the format for the initial data structure:

```
dota_images
├── train
│   ├──images
│   │   └── P0003.png
│   └── labelTxt
│       └── P0003.txt
├── val
│   ├── images
│   └── labelTxt
└── test
    └── images
```

### Annotation format
In the dataset, each object is annotated by an __oriented bounding box (`OBB`)__, which can be denoted as () , where () denotes the i-th vertice of `OBB`. The _vertices_ are arranged in a clockwise order. The following is the visualization of annotations. The yellow point represents the starting point, which means: (a) top-left of a plane, (b) top-left corner of a large vehicle diamond, (c) center of a baseball diamond.

Apart from `OBB`, each instance is also labeled with a category and a difficult which indicates whether the instance is difficult to be detected (1 for difficult, 0 for not difficult). Annotations for an image are saved in a text file with the same file name. Each line represent an instance. The following is an example annotation for an image:

```
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
x1, y1, x2, y2, x3, y3, x4, y4, category, difficult
...
```
>source: https://captain-whu.github.io/DOTA/dataset.html

### Dataloader Dataset Structure

the `dotatron_loader()` in the `dota2kit_data_tools.py` is utilized to load the data in the following format:

```python
data={
    "file_name": filename,  # filename path
    "image_id": idx,        # unique id per object instance
    "height": height,		# height of image
    "width": width,			# width of image
    "annotations":[			# list of annotation(s) for the image_id
        {
            "bbox": [cx,cy,w,h,a]	# stored bbox structure
        	"bbox_mode": BoxMode.	# the detectron2 mode mode enumeration
        	"category_id": 
        }
        {
            "bbox": [cx,cy,w,h,a]
        	"bbox_mode": BoxMode.
        	"category_id": 
        }
    ],
}
```

The BoxMode integer utilized for this dataset is `XYWHA_ABS = 4` which is defined as:
* `(xc, yc, w, h, a)` in absolute floating points coordinates.
* `(xc, yc)` is the center of the rotated box, and the angle a is in degrees ccw.

