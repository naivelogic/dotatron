from detectron2.config import get_cfg
from detectron2 import model_zoo


def base_dota_cfg():

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") # Let training initialize from model zoo

    cfg.DATASETS.TRAIN = (["Train"])
    cfg.DATASETS.TEST = () 
    cfg.TEST.EVAL_PERIOD = 0 # 50 #2000 #x0

    cfg.INPUT.MIN_SIZE_TRAIN = (800,) #(640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 800

    MAX_ITER = 200
    cfg.SOLVER.MAX_ITER = MAX_ITER 
    cfg.SOLVER.STEPS = (.12 *MAX_ITER, .35 *MAX_ITER, .75 * MAX_ITER, .88 * MAX_ITER, .93 * MAX_ITER)  #decay learning rate
    cfg.MODEL.MASK_ON=False
    cfg.MODEL.PROPOSAL_GENERATOR.NAME = "RRPN"
    cfg.MODEL.RPN.HEAD_NAME = "StandardRPNHead"
    cfg.MODEL.RPN.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ANCHOR_GENERATOR.NAME = "RotatedAnchorGenerator"
    cfg.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90,-60,-30,0,30,60,90]]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 #0.5 #0.8 
    cfg.MODEL.ROI_HEADS.NAME = "RROIHeads"
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256 #512
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_name_list)
    cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignRotated"
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10,10,5,5,1)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CONV=4
    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV=8
    cfg.SOLVER.IMS_PER_BATCH = 4 #4 #10 # reduce for memory
    cfg.SOLVER.CHECKPOINT_PERIOD=1500
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.GAMMA=0.5
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.VIS_PERIOD = 10 #50

    #cfg.SOLVER.WARMUP_ITERS = int(0.5 * cfg.SOLVER.MAX_ITER)
    #cfg.SOLVER.WARMUP_FACTOR = 1.0 / (cfg.SOLVER.WARMUP_ITERS + 1)
    #cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0

    cfg.DATALOADER.NUM_WORKERS = 4 #6 #16
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True 
    cfg.DATALOADER.SAMPLER_TRAIN= "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD=0.01
    cfg.MODEL.BACKBONE.FREEZE_AT=6

    return cfg
