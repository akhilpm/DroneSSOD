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
from croptrain.data.datasets.dota import get_overlapping_sliding_window
from utils.box_utils import compute_one_stage_clusters, bbox_scale
from croptrain.data.detection_utils import read_image
from croptrain.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.utils.logger import log_every_n_seconds
logging.basicConfig(level = logging.INFO)
from croptrain.data.datasets.dota import register_dota
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.evaluation import COCOEvaluator
from utils.plot_utils import plot_detections

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

def plot_detection_boxes(predictions, cluster_boxes, data_dict):
    img = Image.open(data_dict["file_name"])
    plt.axis('off')
    plt.imshow(img)
    ax = plt.gca()    
    if len(predictions)!=0:
        predictions = predictions[predictions.scores>0.6]
        predictions = predictions.pred_boxes.tensor.cpu()
        for bbox in predictions:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
    if len(cluster_boxes)!=0:
        if isinstance(cluster_boxes, Instances):
            cluster_boxes = cluster_boxes.pred_boxes.tensor.cpu()
        for bbox in cluster_boxes:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    im_name = os.path.basename(data_dict["file_name"])[:-4]
    plt.savefig(os.path.join('./temp', im_name+"_det.jpg"), dpi=90, bbox_inches='tight')
    #plt.show()
    plt.clf()



def compute_crops_with_prediction(inputs, outputs, cfg):
    instances = outputs[0].get("instances", [])
    instances = instances[instances.scores>0.6]
    crop_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    crop_class_indices = (instances.pred_classes==crop_class)
    instances = instances[~crop_class_indices]
    gt_boxes = instances.pred_boxes.tensor.cpu().numpy().astype(np.int32)
    gt_classes = instances.pred_classes.cpu().numpy().astype(np.int32)
    scaled_boxes = bbox_scale(gt_boxes.copy(), inputs['height'], inputs['width'])
    seg_areas = Boxes(gt_boxes).area()
    data_dict_this_image = copy.deepcopy(inputs)
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


def inference_crops(model, cfg):
    dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)
    print("len of dataset dicts: {}".format(len(dataset_dicts)))
    dataset_name = cfg.DATASETS.TRAIN[0].split("_")[0]
    crop_file = os.path.join(sys.path[0], "dataseed", dataset_name + "_crops_algo_{}.txt".format(cfg.DATALOADER.SUP_PERCENT))
    crop_storage = {}

    cluster_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        count = 0
        n_crops = 0
        for idx, inputs in enumerate(dataset_dicts):
            new_boxes = get_overlapping_sliding_window(dataset_dicts[idx])
            new_data_dicts = get_dict_from_crops(new_boxes, dataset_dicts[idx], cfg.INPUT.MIN_SIZE_TEST)
            image_shapes = [(dataset_dicts[idx].get("height"), dataset_dicts[idx].get("width"))]
            boxes = torch.zeros(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES*4).to(model.device)
            scores = torch.zeros(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1).to(model.device)
            for data_dict in new_data_dicts:
                boxes_patch, scores_patch = model([data_dict], infer_on_crops=True, cfg=cfg)
                boxes = torch.cat([boxes, boxes_patch[0]], dim=0)
                scores = torch.cat([scores, scores_patch[0]], dim=0)
            pred_instances, _ = fast_rcnn_inference([boxes], [scores], image_shapes, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, \
                                    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, cfg.CROPTEST.DETECTIONS_PER_IMAGE)
            pred_instances = pred_instances[0]
            outputs = [{"instances": pred_instances}]
            file_name = dataset_dicts[idx]["file_name"].split('/')[-1]
            if file_name not in crop_storage:
                crop_storage[file_name] = []
            #try:
            crop_boxes = compute_crops_with_prediction(dataset_dicts[idx], outputs, cfg)[:10]
            #except:
            #    print("failed for image {}".format(idx+1))
            #    crop_boxes = []
            if idx%100==0:
                print("processing {}th image".format(idx))
                plot_detection_boxes(pred_instances, crop_boxes, dataset_dicts[idx])
            if len(crop_boxes)>0:
                if not dataset_dicts[idx]["full_image"]:
                    crop_boxes = shift_crop_boxes(dataset_dicts[idx], crop_boxes)
                crop_storage[file_name] += crop_boxes.tolist()
                count += 1
                n_crops += len(crop_boxes)
        del boxes, scores, new_data_dicts        
    print("crops present in {}/{} images".format(count, len(dataset_dicts)))
    print("number of crops is {} ".format(n_crops))
    with open(crop_file, "w") as f:
        json.dump(crop_storage, f)


def main():
    cfg = get_cfg()
    add_croptrainer_config(cfg)
    add_ubteacher_config(cfg)
    cfg.merge_from_file(os.path.join('/home/akhil135/PhD/DroneSSOD', 'configs', 'dota', 'Semi-Sup-RCNN-FPN-CROP.yaml'))
    if cfg.CROPTRAIN.USE_CROPS:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES += 1       
    data_dir = os.path.join(os.environ['SLURM_TMPDIR'], "DOTA")
    dataset_name = cfg.DATASETS.TRAIN[0]
    #cfg.OUTPUT_DIR = "/home/akhil135/scratch/DroneSSOD/DOTA_CROP_SS_10_LR_02"
    #cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/FPN_CROP_SS_1_07/model_0062999.pth" # mAP= 16.74
    #cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/DOTA_CROP_SS_10_LR_02/model_0092999.pth"
    #cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/DOTA_CROP_SS_5/model_0062999.pth"
    cfg.MODEL.WEIGHTS = "/home/akhil135/scratch/DroneSSOD/DOTA_CROP_SS_1_06/model_0020999.pth"
    if not dataset_name in DatasetCatalog:
        #register_visdrone(dataset_name, data_dir, cfg, False)
        register_dota(dataset_name, data_dir, cfg, False)
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
    inference_crops(ensem_ts_model.modelTeacher, cfg)


if __name__ == "__main__":
    main()