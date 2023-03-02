from croptrain.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from croptrain.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from croptrain.modeling.proposal_generator.rpn import PseudoLabRPN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from croptrain import add_croptrainer_config, add_ubteacher_config
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from croptrain.data.datasets.visdrone import register_visdrone
from croptrain.engine.trainer import UBTeacherTrainer, BaselineTrainer
import numpy as np
import torch
import datetime
import time
import copy
import cv2
import json
from utils.crop_utils import get_dict_from_crops
from contextlib import ExitStack, contextmanager
from detectron2.structures.instances import Instances
from detectron2.structures.boxes import Boxes
import matplotlib.pyplot as plt
import logging
from croptrain.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.utils.logger import log_every_n_seconds
from croptrain.data.datasets.visdrone import compute_crops
logging.basicConfig(level = logging.INFO)


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def inference_crops(model, data_loader, cfg):
    #dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
    dataset_name = cfg.DATASETS.TRAIN[0].split("_")[0]
    crop_file = os.path.join("dataseed", dataset_name + "_crops_{}.txt".format(cfg.DATALOADER.SUP_PERCENT))
    crop_storage = {}

    total = len(data_loader)  # inference data loader must have a fixed length
    cluster_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        count = 0
        n_crops = 0
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            cluster_class_indices = (outputs[0]["instances"].pred_classes==cluster_class)
            cluster_boxes = outputs[0]["instances"][cluster_class_indices]
            cluster_boxes = cluster_boxes[cluster_boxes.scores>0.4]
            file_name = inputs[0]["file_name"].split('/')[-1]
            if idx%100==0:
                print("processing {}th image".format(idx))
            if len(cluster_boxes)>0:
                cluster_boxes = cluster_boxes.pred_boxes.tensor.cpu().numpy().astype(np.int32).tolist()
                crop_storage[file_name] = cluster_boxes
                count += 1
                n_crops += len(cluster_boxes)
            else:
                crop_storage[file_name] = []
    with open(crop_file, "w") as f:
        json.dump(crop_storage, f)
    print("crops present in {}/{} images".format(count, len(data_loader)))
    print("number of crops is {} ".format(n_crops))


def main():
    cfg = get_cfg()
    add_croptrainer_config(cfg)
    add_ubteacher_config(cfg)
    cfg.merge_from_file(os.path.join(os.getcwd(), 'configs', 'visdrone', 'Semi-Sup-RCNN-FPN-CROP.yaml'))
    if cfg.CROPTRAIN.USE_CROPS:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES += 1
        cfg.MODEL.RETINANET.NUM_CLASSES += 1
    data_dir = os.path.join(os.environ['SLURM_TMPDIR'], "VisDrone")
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.OUTPUT_DIR = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_10_06"
    cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_10_06/model_0069999.pth"
    if not dataset_name in DatasetCatalog:
        register_visdrone(dataset_name, data_dir, cfg, False)
    if cfg.SEMISUPNET.USE_SEMISUP:
        Trainer = UBTeacherTrainer
    else:
        Trainer = BaselineTrainer

    model = Trainer.build_model(cfg)
    if cfg.SEMISUPNET.USE_SEMISUP:
        model_teacher = Trainer.build_model(cfg)
        ensem_ts_model = EnsembleTSModel(model_teacher, model)
        DetectionCheckpointer(
            ensem_ts_model, save_dir=cfg.OUTPUT_DIR
        ).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
    else:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)


    data_loader = Trainer.build_test_loader(cfg, dataset_name)
    inference_crops(ensem_ts_model.modelTeacher, data_loader, cfg)



if __name__ == "__main__":
    main()