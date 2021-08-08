import os
from PIL import Image
import json
import pickle

from polygon_helper import poly2xywha
from dota_project import DOTA2CLASSES
from detectron2.structures import BoxMode

def dotatron_loader(dataset_dir, load_dataset=False, classlist=DOTA2CLASSES, dataset_load_name="dota2_ds"):

    # cachedir caches the annotations in a pickle file
    save_dataset_path = os.path.join(dataset_dir, dataset_load_name + ".pkl")
    if load_dataset and os.path.exists(save_dataset_path):
        return pickle.load(open(save_dataset_path, "rb"))
        
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labelTxt")

    classlist_flag = True if len(classlist) < len(DOTA2CLASSES) else False

    data_dict = list()
    for path in os.listdir(label_dir):
        filename = os.path.basename(path)
        (file, ext) = os.path.splitext(filename)
        if ext != ".txt":
            continue

        image_filename = file + ".png"
        image_path= os.path.join(image_dir, image_filename)
        file_size = round(os.stat(image_path).st_size / 1000000,1) #st_size: int  # size of file, in bytes,
        if file_size > 40:
        #if file_size > 0.7:
            #print("excluding image bc >30MB: ", image_filename)
            continue

        img = Image.open(image_path)
        height, width = img.size

        image_dict = {
            "file_name": image_path,
            "image_id": file,
            "height": height,
            "width": width,
            "annotations": []
        }


       
        # Create annotations from the labelTxt
        with open(os.path.join(label_dir, filename), 'r') as fp:
            for _, labeltxt_file in enumerate(fp): 
                labeltxt_line = labeltxt_file.strip().split(" ")
                tmp_ann = dict()
                # check for incorrect name
                if len(labeltxt_line) < 9:
                    continue
                if len(labeltxt_line) >= 9:
                    tmp_ann["name"] = labeltxt_line[8]
                if len(labeltxt_line) == 9:
                    tmp_ann["difficult"] = "0"
                elif len(labeltxt_line) >= 10:
                    tmp_ann["difficult"] = labeltxt_line[9]
                
                tmp_ann["poly"] = [float(labeltxt_line[i]) for i in range(8)] # poly format > x1, y1, x2, y2, x3, y3, x4, y4,              

                if classlist_flag and tmp_ann["name"] not in classlist:
                    continue

                annotation = {
                    "category_id": classlist.index(tmp_ann["name"]),
                    "bbox": poly2xywha(tmp_ann["poly"]),
                    "bbox_mode": BoxMode.XYWHA_ABS,
                }
                image_dict["annotations"].append(annotation)

        if len(image_dict["annotations"]) > 0:
            data_dict.append(image_dict)
    
    # save annotaitons to file
    pickle.dump(data_dict, open(save_dataset_path, "wb"))
    return data_dict

