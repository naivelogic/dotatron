from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

def load_dotatron_model(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_filename)
    cfg.MODEL.WEIGHTS = args.ckpt_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.conf
    cfg.MODEL.DEVICE = 'cuda:0'
    if not args.use_gpu:
        cfg.MODEL.DEVICE = 'cpu'  # since we have a training job running
    predictor = DefaultPredictor(cfg)
    return predictor
