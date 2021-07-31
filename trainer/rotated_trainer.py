
import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import RotatedCOCOEvaluator, DatasetEvaluators
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from dota_mapper import DotaDevkitMapper as rotated_mapper

class RotatedTrainer(DefaultTrainer):
    """
    The trainer for rotated box detection task
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluators = RotatedCOCOEvaluator(dataset_name, cfg, True, output_folder)
        return DatasetEvaluators(evaluators)
      
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=rotated_mapper)


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=rotated_mapper)
