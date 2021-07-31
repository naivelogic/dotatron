import os
import copy
import numpy as np
from detectron2.structures import BoxMode
from detectron2.structures.rotated_boxes import RotatedBoxes
from detectron2.data import detection_utils
from detectron2.data import transforms as T
import torch.utils.data

def DotaDevkitMapper(dataset_dict, is_train=True):
    image_format = "BGR"
    dataset_dict = copy.deepcopy(dataset_dict)
    
    image_np = detection_utils.read_image(dataset_dict["file_name"], format=image_format)
    #detection_utils.check_image_size(dataset_dict, image_np)
    
    random_cropper = dotakit_random_crop_instances(dataset_dict, image_np)
    # Data Augmentation 
    #augs = T.AugmentationList([
    #    T.RandomBrightness(0.9, 1.1),
    #    T.RandomFlip(prob=0.5),
    #    T.RandomCrop("absolute", (640, 640))
    # ])
    """
    #SHORT_EDGE = short_edge_length=(800) #short_edge_length=(500 ,640, 672, 800)
    if random_cropper is None:
        tfm_gens = [T.ResizeShortestEdge(short_edge_length=(800), max_size=800, sample_style='choice')]
        #return None
    else:
        #tfm_gens = [random_cropper, T.ResizeShortestEdge(short_edge_length=(800), max_size=800, sample_style='choice'), T.RandomFlip()]
        tfm_gens = [random_cropper, T.Resize((800,800))]
    """
    random_cropper = dotakit_random_crop_instances(dataset_dict, image_np)
    if random_cropper is None:
        tfm_gens = [T.ResizeShortestEdge(short_edge_length=(500 ,640, 672, 800), max_size=800, sample_style='choice'), T.RandomFlip()]
    else:
        tfm_gens = [random_cropper, T.ResizeShortestEdge(short_edge_length=(640, 672, 800), max_size=800, sample_style='choice'), T.RandomFlip()]
    
    
    image, transforms = T.apply_transform_gens(tfm_gens, image_np)
    image_shape = image.shape[:2]  # h, w
    #print("image_shape: ", image_shape)

    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    if not is_train:
        dataset_dict.pop("annotations", None)
        return dataset_dict

    if "annotations" in dataset_dict:
        annos = [
            dota_transform_instance(obj, transforms)
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        instances = dota_annotations_to_instances(annos, image_shape)
        #dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
        dataset_dict["instances"] = instances

    return dataset_dict


def dotakit_random_crop_instances(dataset_dict, image_np):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.
    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """

    image_scale = np.random.uniform(0.09, 2) #2# 1 #original_gsd/target_gsd
    target_size = 800 #dataset_dict['height'] #800
    target_crop = int(target_size / image_scale)
    target_crop = (target_crop, target_crop)
    
    boxes = np.asarray([anno['bbox'] for anno in dataset_dict['annotations']])

    # if some instance is cropped extend the box
    crop_modification = 0
    while True:
        random_crop = T.RandomCrop('absolute', target_crop).get_transform(image_np)
        cropped_boxes = RotatedBoxes(random_crop.apply_coords(copy.deepcopy(boxes)))
        inside_ind = cropped_boxes.inside_box(target_crop, boundary_threshold=5)
        if 1 < sum(inside_ind) <= 100:
            #print("Break")
            return random_crop
            
        crop_modification+=1
        if crop_modification > 100:
            """
            raise ValueError(
                    "Cannot finished cropping adjustment within 100 tries (#instances {}).".format(
                        len(inside_ind)
                    )
                )
            """
            #return T.CropTransform(0, 0, image_size[1], image_size[0])
            #return T.CropTransform(0, 0, 800, 800)
            return None
        

def dota_annotations_to_instances(annos, image_size):
    instances = detection_utils.annotations_to_instances_rotated(annos, image_size)
    instances = detection_utils.filter_empty_instances(instances)
    BIGGER_OBJECTS_TO_EXTEND = [3, 1, 7] #'tennis-court baseball-diamond ground-track-field'
    if np.array(instances.gt_classes).any() in BIGGER_OBJECTS_TO_EXTEND:
        inside_ind = instances.gt_boxes.inside_box(image_size, boundary_threshold=60)
    else: 
        inside_ind = instances.gt_boxes.inside_box(image_size, boundary_threshold=5)
    instances = instances[inside_ind]
    assert ((instances.gt_boxes.tensor.numpy()[:, 2] > 0).all().item()), "width not > 0\n\n" + str(instances.gt_boxes.tensor.numpy())

    return instances


def dota_transform_instance(annotation, transforms):
    """
      Apply transforms to box, segmentation and keypoints annotations of a single instance.
      It will use `transforms.apply_box` for the box, and
      `transforms.apply_coords` for segmentation polygons & keypoints.
      If you need anything more specially designed for each data structure,
      you'll need to implement your own version of this function or the transforms.
      Args:
          annotation (dict): dict of instance annotations for a single instance.
              It will be modified in-place.
          transforms (TransformList or list[Transform]):
          image_size (tuple): the height, width of the transformed image
          keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
      Returns:
          dict:
              the same input dict with fields "bbox", "segmentation", "keypoints"
              transformed according to `transforms`.
              The "bbox_mode" field will be set to XYXY_ABS.
    """
    # updating detectron2/detection_utils transform_instance_annotations function
    # https://github.com/facebookresearch/detectron2/blob/52c81d75817814d66b275313c0325abfcec0a8ca/detectron2/data/detection_utils.py#L257
    
    if annotation["bbox_mode"] == BoxMode.XYWHA_ABS:
        annotation["bbox"] = transforms.apply_rotated_box(np.asarray([annotation["bbox"]]))[0]
    else:
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS
    
    return annotation


def crop_rotated_box(transform, rotated_boxes):
 return transform.apply_coords(rotated_boxes)

T.CropTransform.register_type('rotated_box', crop_rotated_box)