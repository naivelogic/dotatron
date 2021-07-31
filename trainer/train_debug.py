#!/usr/bin/python

import os
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from detectron2.config import *
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import launch

from rotated_trainer import RotatedTrainer
from dota2kit_data_tools import dotatron_loader

from detectron2.engine import default_setup
import logging
logger = logging.getLogger("detectron2")

torch.cuda.set_device(0) # multi 

from cfg_lib import base_dota_cfg
from dota_project import DOTA2CLASSES

def main(args):
  #DATA_FOLDER = "/mnt/omreast_users/phhale/open_ds/DOTA_aerial_images/"
  OUTPUT_PATHS="/home/redne/LUIA_challenge_dev/tron_dota/prod_v2/output/x3/x1"
  TRAIN_PATH = '/home/redne/LUIA_challenge_dev/repos/DOTA_devkit_YOLO/examplesplit'
  
  DatasetCatalog.clear()
  MetadataCatalog.clear()

  DatasetCatalog.register("Train", lambda: dotatron_loader(dataset_dir=TRAIN_PATH, load_dataset=False, classlist=DOTA2CLASSES, dataset_load_name='train_dota2kit_prodv2'))
  MetadataCatalog.get("Train").set(thing_classes=DOTA2CLASSES)
  

  cfg = base_dota_cfg()
  cfg.DATASETS.TRAIN = (["Train"])
  cfg.DATASETS.TEST = () 
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(DOTA2CLASSES)
  cfg.VIS_PERIOD = 10 #50
  cfg.OUTPUT_DIR= OUTPUT_PATHS 
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) #lets just check our output dir exists

  cfg.freeze()                    # make the configuration unchangeable during the training process
  default_setup(cfg, args)
  cfg.dump()
  
  trainer = RotatedTrainer(cfg) 
  #trainer.resume_or_load(resume=True)
  trainer.resume_or_load(resume=False)
  return trainer.train()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    from custom_arg_parsers import custom_default_argument_parser
    args = custom_default_argument_parser().parse_args()
    print("Command Line Args:", args)
    torch.backends.cudnn.benchmark = True
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
