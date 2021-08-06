MNT_ROOT=/mnt/omreast_users/phhale/open_ds/DOTA_aerial_images

#INPUT_DIR=/home/redne/LUIA_challenge_dev/repos/DOTA_devkit_YOLO/examplesplit/images/
INPUT_DIR=${MNT_ROOT}/images/dev_demo/dota2_small_set_v2/dota2_split200/val/images/
OUTPUT_DIR=/home/redne/LUIA_challenge_dev/tron_dota/ws/dota2_split200_val/random
CONF=0.5
USE_GPU=True
CKPT_PATH=${MNT_ROOT}/experiments/detectron2/080421_dotatrainval_devkit_frcnn_X101_nc24x01/model_0019999.pth
NUM_IMAGES=20
python test_tron.py --input_dir ${INPUT_DIR} --output_dir ${OUTPUT_DIR} --conf ${CONF} \
                    --use_gpu ${USE_GPU} --ckpt_path ${CKPT_PATH} --num_images ${NUM_IMAGES}