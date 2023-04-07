import sys
sys.path.insert(0, '/home/akhil135/PhD/DroneSSOD')
from croptrain.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN
from croptrain.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
from croptrain.modeling.proposal_generator.rpn import PseudoLabRPN
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from croptrain import add_croptrainer_config, add_ubteacher_config
from detectron2.data import DatasetCatalog, MetadataCatalog
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
from utils.crop_utils import get_dict_from_crops
from utils.box_utils import compute_one_stage_clusters, bbox_scale
from croptrain.data.detection_utils import read_image
from croptrain.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.utils.logger import log_every_n_seconds
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

def shift_crop_boxes(data_dict, cluster_boxes):
    x1, y1 = data_dict["crop_area"][0], data_dict["crop_area"][1]
    ref_point = np.array([x1, y1, x1, y1])
    cluster_boxes = cluster_boxes + ref_point
    return cluster_boxes


def compute_crops_with_prediction(inputs, outputs, cfg):
    instances = outputs[0].get("instances", [])
    instances = instances[instances.scores>0.6]
    crop_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    crop_class_indices = (instances.pred_classes==crop_class)
    instances = instances[~crop_class_indices]
    gt_boxes = instances.pred_boxes.tensor.cpu().numpy().astype(np.int32)
    gt_classes = instances.pred_classes.cpu().numpy().astype(np.int32)
    scaled_boxes = bbox_scale(gt_boxes.copy(), inputs[0]['height'], inputs[0]['width'])
    seg_areas = Boxes(gt_boxes).area()
    data_dict_this_image = copy.deepcopy(inputs[0])
    objs = []
    for i in range(len(gt_boxes)):
        obj = {}
        obj["bbox"] = gt_boxes[i].tolist()
        obj["category_id"] = gt_classes[i]
        objs.append(obj)
    data_dict_this_image["annotations"] = objs    
    #stage 1 - merging
    data_dict_this_image, new_boxes, new_seg_areas = compute_one_stage_clusters(data_dict_this_image, scaled_boxes, seg_areas, cfg, stage=1)
    #stage 2 - merging
    data_dict_this_image, new_boxes, new_seg_areas = compute_one_stage_clusters(data_dict_this_image, new_boxes, new_seg_areas, cfg, stage=2)
    return new_boxes


def inference_crops(model, data_loader, cfg):
    #dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
    dataset_name = cfg.DATASETS.TRAIN[0].split("_")[0]
    crop_file = os.path.join(sys.path[0], "dataseed", dataset_name + "_crops_algo_{}.txt".format(cfg.DATALOADER.SUP_PERCENT))
    crop_storage = {}

    total = len(data_loader)  # inference data loader must have a fixed length
    cluster_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        count = 0
        n_crops = 0
        for idx, inputs in enumerate(data_loader):
            outputs = model(inputs)
            file_name = inputs[0]["file_name"].split('/')[-1]
            if file_name not in crop_storage:
                crop_storage[file_name] = []
            crop_boxes = compute_crops_with_prediction(inputs, outputs, cfg)    
            if idx%300==0:
                print("processing {}th image".format(idx))
            if len(crop_boxes)>0:
                if not inputs[0]["full_image"]:
                    crop_boxes = shift_crop_boxes(inputs[0], crop_boxes)
                crop_storage[file_name] += crop_boxes.tolist()
                count += 1
                n_crops += len(crop_boxes)
    print("crops present in {}/{} images".format(count, len(data_loader)))
    print("number of crops is {} ".format(n_crops))
    with open(crop_file, "w") as f:
        json.dump(crop_storage, f)


def main():
    cfg = get_cfg()
    add_croptrainer_config(cfg)
    add_ubteacher_config(cfg)
    cfg.merge_from_file(os.path.join('/home/akhil135/PhD/DroneSSOD', 'configs', 'visdrone', 'Semi-Sup-RCNN-FPN-CROP.yaml'))
    if cfg.CROPTRAIN.USE_CROPS:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES += 1
    data_dir = os.path.join(os.environ['SLURM_TMPDIR'], "VisDrone")
    dataset_name = cfg.DATASETS.TRAIN[0]
    cfg.OUTPUT_DIR = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_1"
    #cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_5/model_0047999.pth" # mAP = 22.52
    cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_1_07/model_0062999.pth" # mAP= 16.74
    #cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_10_06/model_0071999.pth" # mAP = 26.48
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