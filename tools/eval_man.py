import os, sys
import cv2
import numpy as np
import random 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

sys.path.append("datasets/dota_devkit")
from tools.dota2_defaults import DOTA2CLASSES, DEFAULT_DOTA2_COLORS
from tools.test_model import load_dotatron_model

sys.path.append("/home/redne/dotatron/")
sys.path.append("/home/redne/dotatron/datasets/dota_devkit/")
from datasets.dota_devkit.ResultMerge import mergebypoly
from datasets.dota_devkit.dota_utils import custombasename, GetFileFromThisRootDir
from datasets.dota_devkit.dota_evaluation_task1 import voc_eval


def poly2xywha(cx, cy, width, height, theta):
    """
    Check the angle in the OPENCV format for problems and record and change them
    """
    if theta == 0:
        theta = -90
        tmp = width
        width = height
        height = tmp
    
    if width != max(width, height):
        # width is not the longest edge
        theta = theta - 90
        return cx, cy, height, width, theta
    else:
        # width is the longest edge 
        return cx, cy, width, height, theta


def make_new_folder(folder):
    os.makedirs(folder, exist_ok=True)
    #if not os.path.exists(folder):
    #    os.makedirs(folder)  # make new output folder

def save_txt_result_before_merge(lines, base_img_name, out_path):
    with open(str(out_path + '/' + base_img_name) + '.txt', 'a') as f:
        f.writelines(lines + '\n')

def detect_img(predictor, test_img, out_path):
    #print(test_img)
    im = cv2.imread(test_img)
    im = im[:, :, ::-1]
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    all_outputs = outputs['instances'].to("cpu")
    boxes = all_outputs.get_fields()['pred_boxes'].tensor.numpy().astype(float)
    scores = all_outputs.get_fields()['scores'].numpy().astype(float)
    pred_classes = all_outputs.get_fields()['pred_classes'].numpy().astype(int, casting='safe')

    imgname = os.path.basename(test_img)[:-4]
    base_img_name = imgname.split('__')[0] #['P1088', '2', '924', '_2772']

    for pred_class, rot_boxe, score in zip(pred_classes, boxes, scores):
        r1 = poly2xywha(rot_boxe[0], rot_boxe[1], rot_boxe[2], rot_boxe[3],-1*rot_boxe[4])

        rect = ((r1[0], r1[1]), (r1[2], r1[3]),r1[4])
        poly = np.float32(cv2.boxPoints(rect))
        ppoly = np.int0(poly).reshape(8)
        lines = str(imgname) + ' ' + str(round(score,3)) + ' ' + ' '.join(list(map(str, ppoly))) + ' ' + str(DOTA2CLASSES[pred_class])
        save_txt_result_before_merge(lines=lines, base_img_name=base_img_name, out_path=out_path)


def get_image_dir_list(im_dir):
    return os.listdir(im_dir)

def merge_detection_splits(detect_split_dir, det_merge_dir):
    # Step #2 to Merge the detection results from Image Splits after model detection
    ## save_txt_result_after_merge
    print('>> Merging the detections results on the split images')
    print(f'>> Using splits from {detect_split_dir}')
    mergebypoly(detect_split_dir, det_merge_dir)
    print(f'>> Detection Results on full images: {det_merge_dir}')

def save_txt_result_after_merge_by_class(det_merge_dir, det_class_dir):
    list_of_imgs = GetFileFromThisRootDir(det_merge_dir)

    for file in list_of_imgs:
        with open(file, 'r') as orig_file:
            lines = orig_file.readlines()
            line = [x.strip().split(" ") for x in lines]
            for l in line:
                class_file_name = 'Task1_' + l[-1]
                strline = ' '.join(list(l[:-1]))
                save_txt_result_before_merge(lines=strline, base_img_name = class_file_name, out_path = det_class_dir)


    print(f">> Task 1 Submission by Class results path: {det_class_dir}")
            
def save_imagelist_txt(img_dir ,gt_images_file):
    # Step 3 create txt file of list of images
    list_of_imgs = get_image_dir_list(img_dir)
    for img_file in list_of_imgs:
        with open(gt_images_file, 'a') as f:
            f.writelines(custombasename(img_file) + '\n')
    print(f">> List of Images saved: {gt_images_file}")

def map_eval(result_classname_path,gt_anno_dir,gt_images_file,classnames=DOTA2CLASSES):
    # task1 dota_Devkit helder
    detpath = str(result_classname_path + '/Task1_{:s}.txt')  
    annopath = gt_anno_dir
    imagesetfile = gt_images_file

    
    classaps = []
    map = 0
    no_det_class_counter = 0
    for classname in classnames:
        print('classname:', classname)
        detfile = detpath.format(classname)
        if not (os.path.exists(detfile)):
            no_det_class_counter += 1
            print('No detections for class: {:s}'.format(classname))
            continue
        
        rec, prec, ap = voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.5,
             use_07_metric=True)
        map = map + ap
        print('ap: ', ap)
        classaps.append(ap)

    map = map/(len(classnames)-no_det_class_counter)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)

def draw_merge_results(orig_img_dir, merge_det_dir, merge_vis_dir, classnames):
    make_new_folder(merge_vis_dir)

    colors = DEFAULT_DOTA2_COLORS
    det_file_list = GetFileFromThisRootDir(merge_det_dir)

    for det_file in det_file_list:
        results = list()
        with open(det_file, 'r') as f:
            lines = f.readlines()
            line = [x.strip().split(" ") for x in lines]
            results = [x[2:-1] for x in line]
            result_class = [x[-1] for x in line]

        name = os.path.splitext(os.path.basename(det_file))[0]  
        orig_image = os.path.join(orig_img_dir, name + '.png')
        save_image = os.path.join(merge_vis_dir, name + '.png')
        img = cv2.imread(orig_image)  

        for i, obj in enumerate(results):
            result_cls = result_class[i]
            poly = np.array(list(map(float, obj)))
            poly = poly.reshape(4, 2)  
            poly = np.int0(poly)

            cv2.drawContours(image=img,
                             contours=[poly],
                             contourIdx=-1,
                             color=colors[int(classnames.index(result_cls))],
                             thickness=2)
        cv2.imwrite(save_image, img)

def eval_folders_buildout(args):
    # 0. Set up Folders and Arguments
    # setup_output_folder
    model_ckp_name = os.path.basename(args.ckpt_path).split('_')[1][:-4] #['model', '0005999.pth'] #'0005999'
    
    #save path for results
    args.output_dir = os.path.join(args.output_dir, model_ckp_name)
    make_new_folder(args.output_dir) 
    
    det_splits_path = os.path.join(args.output_dir ,'imgsplits_results')
    make_new_folder(det_splits_path) 

    model_dir = os.path.dirname(args.ckpt_path) # experiments/detectron2/080421_dotatrainval_devkit_frcnn_X101_nc24x01
    model_name = os.path.basename(model_dir)    # '080421_dotatrainval_devkit_frcnn_X101_nc24x01'
    args.config_filename = os.path.join(model_dir, 'config.yaml')

    # 3.0 Eval Folder Set up after detectron on splits
    det_result_dir = args.output_dir
    det_result_annotations = os.path.join(det_result_dir, "result_merged")
    det_class_results = os.path.join(det_result_dir, "results_by_class")
    gt_images_file = os.path.join(args.output_dir, "gt_imagelist.txt")
    det_merge_img_viz_dir = os.path.join(det_result_dir, "merge_vis")
    
    gt_img_dir = args.gt_img_dir 
    gt_img_annotations = args.gt_anno_dir 

    # 2.1 Load the detector model
    predictor = load_dotatron_model(args)

    # 2. Detect Image (Open Imgs > Predict > Save results on image subset (need to merge))
    test_img_list = get_image_dir_list(args.input_dir)
    for test_img_name in tqdm(test_img_list):
        test_img = os.path.join(args.input_dir, test_img_name)
        detect_img(predictor, test_img, det_splits_path)
    

    # step 3
    make_new_folder(det_result_annotations)
    merge_detection_splits(det_splits_path,det_result_annotations)

    # step 4 det_class_results
    make_new_folder(det_class_results)
    save_txt_result_after_merge_by_class(det_result_annotations, det_class_results)

    # step 5.1 > GT create image file lise
    save_imagelist_txt(gt_img_dir, gt_images_file)

    # step 5.2 > run mAP on class detection results
    map_eval(result_classname_path=det_class_results, gt_anno_dir=gt_img_annotations, 
             gt_images_file=gt_images_file, classnames=DOTA2CLASSES)
    
    # step 8
    make_new_folder(det_merge_img_viz_dir)
    draw_merge_results(orig_img_dir=gt_img_dir, merge_det_dir=det_result_annotations,
                       merge_vis_dir=det_merge_img_viz_dir, classnames=DOTA2CLASSES)


if __name__ == "__main__":
    import argparse
    MNT_ROOT="/mnt/omreast_users/phhale/open_ds/DOTA_aerial_images"
    args = argparse.Namespace(
        input_dir = os.path.join(MNT_ROOT, "images/dev_demo/dota2_small_set_v2/dota2_split200/val/images/"),
        gt_img_dir = os.path.join(MNT_ROOT, "images/dev_demo/dota2_small_set_v2/dota2_dataset/val/images/"),
        gt_anno_dir = os.path.join(MNT_ROOT, "images/dev_demo/dota2_small_set_v2/dota2_dataset/val/labelTxt/{:s}.txt"),
        output_dir = "/home/redne/LUIA_challenge_dev/tron_dota/ws/dota2_split200_val",
        conf = 0.6,
        use_gpu = True,
        ckpt_path = os.path.join(MNT_ROOT,"experiments/detectron2/080421_dotatrainval_devkit_frcnn_X101_nc24x01/model_0025999.pth"),
        num_images = 20
    )
    print(args)

    eval_folders_buildout(args)