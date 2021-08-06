# Getting Started with DotaTron

This section is a overview of initiating the detectron2 training and evaluation on oriented rotated datasets challenges. The primary usecase tested in the project is the DOTA-v2 dataset challenge. Refer to the [README.md](../README.md) to prereq and review the [DATASET_Details.md](02_Dataset_Details.md) for data preparation and annotation format details. 


## Training

### Train detection model 

to train run: `TODO`

Makesure these model configs are correct:

```yaml
DATA_LOADER.NUM_WORKERS 0 
NUM_GPUS 2 
TRAIN.BATCH_SIZE 16 
```

__Resume training from checkpoint__

For training that has already started and wanting to retrain an existing model run: `TODO`


__Tensorboard__

```
cd $PATH_MODEL_LOG/outputs
tensorboard --logdir=.
```


## Test and Eval

### Visualize Detection Results

To test the model inference run the following or run `sh scripts/test_train.sh`

```
INPUT_DIR=
OUTPUT_DIR=
CONF=
USE_GPU=
CKPT_PATH=
NUM_IMAGES=20
python test_tron.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --conf ${CONF} \
                    --use_gpu ${USE_GPU} --ckpt_path ${CKPT_PATH} --num_images ${NUM_IMAGES}
```

### Evaluation

To test the model inference run the following or run `sh scripts/test_train.sh`

```

INPUT_DIR=
OUTPUT_DIR=
CONF=
USE_GPU=True
CKPT_PATH=
NUM_IMAGES=20
python test_tron.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --conf ${CONF} \
                    --use_gpu ${USE_GPU} --ckpt_path ${CKPT_PATH} --num_images ${NUM_IMAGES}
```