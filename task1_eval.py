import numpy as np
import cv2
import os, sys
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

sys.path.append("datasets/dota_devkit")
from tools.args_lib import test_tron_args
from tools.test_model import load_dotatron_model

def get_image_dir_list(im_dir):
    return os.listdir(im_dir)


def save_txt_result_before_merge(lines, base_img_name, out_path):
    with open(str(out_path + '/' + base_img_name) + '.txt', 'a') as f:
        f.writelines(lines + '\n')

from tools.eval_man import detect_img

if __name__ == "__main__":
    args = test_tron_args().parse_args()
    print("Command Line Args:", args)

    # 0. Set up Folders and Arguments
    # setup_output_folder
    model_ckp_name = os.path.basename(args.ckpt_path).split('_')[1][:-4] #['model', '0005999.pth'] #'0005999'
    
    #save path for results
    args.output_dir = os.path.join(args.output_dir, model_ckp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    det_splits_path = os.path.join(args.output_dir ,'imgsplits_results')
    if not os.path.exists(det_splits_path):
        os.makedirs(det_splits_path)  # make new output folder

    model_dir = os.path.dirname(args.ckpt_path) # experiments/detectron2/080421_dotatrainval_devkit_frcnn_X101_nc24x01
    model_name = os.path.basename(model_dir) # '080421_dotatrainval_devkit_frcnn_X101_nc24x01'
    args.config_filename = os.path.join(model_dir, 'config.yaml')
    
    # 1. Load the detector model
    predictor = load_dotatron_model(args)
    
    # 2. Detect Image (Open Imgs > Predict > Save results on image subset (need to merge))
    test_img_list = get_image_dir_list(args.input_dir)
    for test_img_name in tqdm(test_img_list):
        test_img = os.path.join(args.input_dir, test_img_name)
        detect_img(predictor, test_img, det_splits_path)
