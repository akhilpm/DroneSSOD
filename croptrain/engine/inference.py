from turtle import width
import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator, DatasetEvaluators
import os
import datetime
import time
import copy
from contextlib import ExitStack, contextmanager
import logging
from torchvision.transforms import Resize
from detectron2.structures.instances import Instances
from utils.box_utils import bbox_inside_old
from utils.plot_utils import plot_detections
from detectron2.utils.logger import log_every_n_seconds
from detectron2.structures.boxes import Boxes, pairwise_iou
from croptrain.data.detection_utils import read_image
from croptrain.data.datasets.visdrone import compute_crops, uniform_cropping
from detectron2.data.build import get_detection_dataset_dicts
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
logging.basicConfig(level=logging.INFO)


def prune_boxes_inside_cluster(cluster_dicts, boxes, scores, num_classes):
    boxes = boxes[0].reshape(-1, 4)
    scores = scores[0].flatten()
    for cluster_dict in cluster_dicts:
        crop_area = cluster_dict["crop_area"]
        inside_boxes = bbox_inside_old(crop_area, boxes)
        boxes[inside_boxes] = 0.0
        scores[inside_boxes] = 0.0
    boxes = boxes.view(-1, num_classes * 4)
    scores = scores.view(-1, num_classes)
    return [boxes], [scores]


def inference_with_crops(model, data_loader, evaluator, cfg, iter):
    from detectron2.utils.comm import get_world_size
    #dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TEST, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS)

    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()
    save_dir = os.path.join(cfg.OUTPUT_DIR, "detections")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    #last_class = evaluator._metadata.get("thing_classes")[-1]
    #if last_class=="cluster":
    #    all_classes = evaluator._metadata.get("thing_classes")
    #    evaluator._metadata.__dict__['thing_classes'] =  all_classes[:-1]
    #    evaluator._metadata.__dict__['thing_dataset_id_to_contiguous_id'].pop(cfg.MODEL.ROI_HEADS.NUM_CLASSES)

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    cluster_class = cfg.MODEL.ROI_HEADS.NUM_CLASSES - 1
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            image_shapes = [(item.get("height"), item.get("width")) for item in inputs]
            outputs = model(inputs)
            cluster_class_indices = (outputs[0]["instances"].pred_classes==cluster_class)
            cluster_boxes = outputs[0]["instances"][cluster_class_indices]
            cluster_boxes = cluster_boxes[cluster_boxes.scores>0.7]

            #_, clus_dicts = compute_crops(dataset_dicts[idx], cfg)
            #cluster_boxes = np.array([item['crop_area'] for item in clus_dicts]).reshape(-1, 4)
            
            if len(cluster_boxes)!=0:
                #cluster_boxes = merge_cluster_boxes(cluster_boxes, cfg)
                cluster_dicts = get_dict_from_crops(cluster_boxes, inputs[0], cfg.CROPTEST.CROPSIZE)
                boxes, scores = model(inputs, cluster_dicts, infer_on_crops=True)
            else:
                boxes, scores = model(inputs, None, infer_on_crops=True)
            pred_instances, _ = fast_rcnn_inference(boxes, scores, image_shapes, cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST, \
                                    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, cfg.CROPTEST.DETECTIONS_PER_IMAGE)
            pred_instances = pred_instances[0]
            pred_instances = pred_instances[pred_instances.pred_classes!=cluster_class]
            all_outputs = [{"instances": pred_instances}]
            
            #if idx%100==0:
            #    plot_detections(pred_instances.to("cpu"), cluster_boxes, inputs[0], evaluator._metadata, cfg, iter)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, all_outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results



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


def get_dict_from_crops(crops, input_dict, CROPSIZE):
    if len(crops)==0:
        return []
    if isinstance(crops, Instances):
        crops = crops.pred_boxes.tensor.cpu().numpy().astype(np.int32)
    transform = Resize(CROPSIZE)
    crop_dicts, crop_scales = [], []
    for i in range(len(crops)):
        x1, y1, x2, y2 = crops[i, 0], crops[i, 1], crops[i, 2], crops[i, 3]
        crop_size_min = min(x2-x1, y2-y1)
        if crop_size_min<=0:
            continue
        crop_dict = copy.deepcopy(input_dict)
        crop_dict['full_image'] = False
        crop_dict['crop_area'] = np.array([x1, y1, x2, y2])
        crop_region = read_image(crop_dict)
        crop_region = torch.as_tensor(np.ascontiguousarray(crop_region.transpose(2, 0, 1)))
        crop_region = transform(crop_region)
        crop_dict["image"] = crop_region
        crop_dict["height"] = (y2-y1)
        crop_dict["width"] = (x2-x1)
        crop_dicts.append(crop_dict)
        crop_scales.append(float(CROPSIZE)/crop_size_min)

    return crop_dicts


def merge_cluster_boxes(cluster_boxes, cfg):
    if len(cluster_boxes)==0:
        return None
    if len(cluster_boxes)==1:
        box = cluster_boxes.pred_boxes.tensor.cpu().numpy().astype(np.int32).reshape(1, -1)
        return box

    overlaps = pairwise_iou(cluster_boxes.pred_boxes, cluster_boxes.pred_boxes)
    connectivity = (overlaps > cfg.CROPTRAIN.CLUSTER_THRESHOLD)
    new_boxes = np.zeros((0, 4), dtype=np.int32)
    while len(connectivity)>0:
        connections = connectivity.sum(dim=1)
        max_connected, max_connections = torch.argmax(connections), torch.max(connections)
        cluster_components = torch.nonzero(connectivity[max_connected]).view(-1)
        other_boxes = torch.nonzero(~connectivity[max_connected]).view(-1)
        if max_connections==1:
            box = cluster_boxes.pred_boxes.tensor[max_connected]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        else:
            cluster_members = cluster_boxes.pred_boxes.tensor[cluster_components]
            x1, y1 = cluster_members[:, 0].min(), cluster_members[:, 1].min()
            x2, y2 = cluster_members[:, 2].max(), cluster_members[:, 3].max()
        crop_area = np.array([int(x1), int(y1), int(x2), int(y2)]).astype(np.int32)
        new_boxes = np.append(new_boxes, crop_area.reshape(1, -1), axis=0)
        connectivity = connectivity[:, other_boxes]
        connectivity = connectivity[other_boxes, :]

    return new_boxes
