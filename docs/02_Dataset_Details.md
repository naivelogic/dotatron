# About the Dota Dataset

* Training Dataset: https://captain-whu.github.io/DOTA/dataset.html

__Overview__
* __Inputs:__ DOTA iamges
* __Input Size:__ 1024 x 1024 x 3 | _dev to experiment: 608 x 608 x 3_
* __Outputs: 7 degrees of freedom (7-DOF)__ of objects: _(cx, cy, cz, l, w, h, θ)_
  * `cx, cy, cz`: the center coordinates
  * `l, w, h`: length, width, height of bounding box
  * `θ`: The heading angle in radians of the bounding box
* __Objects:__ 18 Dota-2.0 classes

Straightens and crops the image using the information of the __oriented rectangle boxes__ in the image, `width`, `height`, `center point` and `rotation degree`.


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
