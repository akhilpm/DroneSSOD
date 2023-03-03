# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import operator
import json
import os
import torch.utils.data
from detectron2.utils.comm import get_world_size
from detectron2.data.common import (
    DatasetFromList,
    MapDataset,
)
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers import (
    InferenceSampler,
    RepeatFactorTrainingSampler,
    TrainingSampler,
)
from detectron2.data.build import (
    trivial_batch_collator,
    worker_init_reset_seed,
    get_detection_dataset_dicts,
    build_batch_data_loader,
)
from croptrain.data.common import (
    AspectRatioGroupedSemiSupDatasetTwoCrop,
)
from croptrain.data.dataset_mapper import DatasetMapperTwoCropSeparate
from croptrain.data.dataset_mapper import DatasetMapperDensityCrop
from utils.crop_utils import get_dict_from_crops

"""
This file contains the default logic to build a dataloader for training or testing.
"""


def divide_label_unlabel(dataset_dicts, cfg):
    dataset_name = cfg.DATASETS.TRAIN[0].split("_")[0]
    seed_file = os.path.join("dataseed", dataset_name + "_sup_{}.txt".format(cfg.DATALOADER.SUP_PERCENT))
    with open(seed_file) as f:
        random_perm_data = json.load(f)
    random_perm = random_perm_data["perm"]
    num_all = len(random_perm)
    num_label = int(cfg.DATALOADER.SUP_PERCENT / 100.0 * num_all)
    shuffled_images = [random_perm_data["imagenames"][x] for x in random_perm]   
    labeled_image_ids = shuffled_images[:num_label]
    if cfg.SEMISUPNET.AUG_CROPS_UNSUP:
        crop_file = os.path.join("dataseed", dataset_name + "_crops_{}.txt".format(cfg.DATALOADER.SUP_PERCENT))
        with open(crop_file) as f:
            crop_data = json.load(f)
    label_dicts = []
    unlabel_dicts = []

    for i in range(len(dataset_dicts)):
        file_name = dataset_dicts[i]["file_name"].split('/')[-1]
        if file_name in labeled_image_ids:
            label_dicts.append(dataset_dicts[i])
        elif dataset_dicts[i]['full_image']:
            unlabel_dicts.append(dataset_dicts[i])
            if cfg.SEMISUPNET.AUG_CROPS_UNSUP:
                crop_boxes = np.array(crop_data[file_name])
                if len(crop_boxes)>0:
                    crop_dicts = get_dict_from_crops(crop_boxes, dataset_dicts[i], with_image=False)
                    unlabel_dicts += crop_dicts
        elif (dataset_dicts[i].get("inner_crop_area")!=None and dataset_dicts[i]["two_stage_crop"]==False):
            unlabel_dicts.append(dataset_dicts[i])
            if cfg.SEMISUPNET.AUG_CROPS_UNSUP:
                crop_boxes = np.array(crop_data[file_name])
                if len(crop_boxes)>0:
                    crop_dicts = get_dict_from_crops(crop_boxes, dataset_dicts[i], with_image=False)
                    unlabel_dicts += crop_dicts
        else:
            continue   
    return label_dicts, unlabel_dicts


def divide_label_unlabel_supervised(dataset_dicts, cfg):
    num_all = len(dataset_dicts)
    num_label = int(cfg.DATALOADER.SUP_PERCENT/100.0*num_all)

    # generate a permutation of images
    random_perm_index = np.random.permutation(num_all)
    label_dicts = []
    unlabel_dicts = []
    for i in range(len(dataset_dicts)):
        if i < num_label:
            label_dicts.append(dataset_dicts[random_perm_index[i]])
        else:
            unlabel_dicts.append(dataset_dicts[random_perm_index[i]])
    return label_dicts, unlabel_dicts

# uesed by supervised-only baseline trainer
def build_detection_semisup_train_loader(cfg, mapper=None):

    dataset_dicts = get_detection_dataset_dicts(
        cfg.DATASETS.TRAIN,
        filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
        min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
        if cfg.MODEL.KEYPOINT_ON
        else 0,
        proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )

    # Divide into labeled and unlabeled sets according to supervision percentage
    label_dicts, unlabel_dicts = divide_label_unlabel_supervised(
        dataset_dicts,
        cfg
    )

    dataset = DatasetFromList(label_dicts, copy=False)

    if cfg.CROPTRAIN.USE_CROPS:
        mapper = DatasetMapperDensityCrop(cfg, True)
    if "dota" in cfg.DATASETS.TRAIN[0] or "dota" in cfg.DATASETS.TEST[0]:
        mapper = DatasetMapperDensityCrop(cfg, True)
    if mapper is None:
        mapper = DatasetMapper(cfg, True)
    dataset = MapDataset(dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))

    if sampler_name == "TrainingSampler":
        sampler = TrainingSampler(len(dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        repeat_factors = (
            RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                label_dicts, cfg.DATALOADER.REPEAT_THRESHOLD
            )
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info(" Number of training samples " + str(len(dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))

    return build_batch_data_loader(
        dataset,
        sampler,
        cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# uesed by evaluation
def build_detection_test_loader(cfg, dataset_name, mapper=None):
    dataset_dicts = get_detection_dataset_dicts(
        [dataset_name],
        filter_empty=False,
        proposal_files=[
            cfg.DATASETS.PROPOSAL_FILES_TEST[
                list(cfg.DATASETS.TEST).index(dataset_name)
            ]
        ]
        if cfg.MODEL.LOAD_PROPOSALS
        else None,
    )
    dataset = DatasetFromList(dataset_dicts)
    if "dota" in cfg.DATASETS.TRAIN[0] or "dota" in cfg.DATASETS.TEST[0]:
        mapper = DatasetMapperDensityCrop(cfg, False)
    if mapper is None:
        mapper = DatasetMapper(cfg, False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


# uesed by unbiased teacher trainer
def build_detection_semisup_train_loader_two_crops(cfg, mapper=None):
    if cfg.DATASETS.CROSS_DATASET:  # cross-dataset (e.g., coco-additional)
        label_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_LABEL,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
        unlabel_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN_UNLABEL,
            filter_empty=False,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )
    else:  # different degree of supervision (e.g., COCO-supervision)
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN
            if cfg.MODEL.LOAD_PROPOSALS
            else None,
        )

        # Divide into labeled and unlabeled sets according to supervision percentage
        label_dicts, unlabel_dicts = divide_label_unlabel(
            dataset_dicts,
            cfg
        )

    label_dataset = DatasetFromList(label_dicts, copy=False)
    unlabel_dataset = DatasetFromList(unlabel_dicts, copy=False)
    print("Labeled samples:{}, Unlabeled samples:{} ".format(len(label_dicts), len(unlabel_dicts)))
    if mapper is None:
        if cfg.CROPTRAIN.USE_CROPS:
            mapper = DatasetMapperTwoCropSeparate(cfg, True)
        else:
            mapper = DatasetMapper(cfg, True)
        if "dota" in cfg.DATASETS.TRAIN[0] or "dota" in cfg.DATASETS.TEST[0]:
            mapper = DatasetMapperTwoCropSeparate(cfg, True)
    label_dataset = MapDataset(label_dataset, mapper)
    unlabel_dataset = MapDataset(unlabel_dataset, mapper)

    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    logger.info("Using training sampler {}".format(sampler_name))
    if sampler_name == "TrainingSampler":
        label_sampler = TrainingSampler(len(label_dataset))
        unlabel_sampler = TrainingSampler(len(unlabel_dataset))
    elif sampler_name == "RepeatFactorTrainingSampler":
        raise NotImplementedError("{} not yet supported.".format(sampler_name))
    else:
        raise ValueError("Unknown training sampler: {}".format(sampler_name))

    # list num of labeled and unlabeled
    logger.info("Number of LABELED training samples " + str(len(label_dataset)))
    logger.info("Number of UNLABELED training samples " + str(len(unlabel_dataset)))
    logger.info("Supervision percentage " + str(cfg.DATALOADER.SUP_PERCENT))
    return build_semisup_batch_data_loader_two_crop(
        (label_dataset, unlabel_dataset),
        (label_sampler, unlabel_sampler),
        cfg.SOLVER.IMG_PER_BATCH_LABEL,
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


# batch data loader
def build_semisup_batch_data_loader_two_crop(
    dataset,
    sampler,
    total_batch_size_label,
    total_batch_size_unlabel,
    *,
    aspect_ratio_grouping=False,
    num_workers=0
):
    world_size = get_world_size()
    assert (
        total_batch_size_label > 0 and total_batch_size_label % world_size == 0
    ), "Total label batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    assert (
        total_batch_size_unlabel > 0 and total_batch_size_unlabel % world_size == 0
    ), "Total unlabel batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size_label, world_size
    )

    batch_size_label = total_batch_size_label // world_size
    batch_size_unlabel = total_batch_size_unlabel // world_size

    label_dataset, unlabel_dataset = dataset
    label_sampler, unlabel_sampler = sampler

    if aspect_ratio_grouping:
        label_data_loader = torch.utils.data.DataLoader(
            label_dataset,
            sampler=label_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        unlabel_data_loader = torch.utils.data.DataLoader(
            unlabel_dataset,
            sampler=unlabel_sampler,
            num_workers=num_workers,
            batch_sampler=None,
            collate_fn=operator.itemgetter(
                0
            ),  # don't batch, but yield individual elements
            worker_init_fn=worker_init_reset_seed,
        )  # yield individual mapped dict
        return AspectRatioGroupedSemiSupDatasetTwoCrop(
            (label_data_loader, unlabel_data_loader),
            (batch_size_label, batch_size_unlabel),
        )
    else:
        raise NotImplementedError("ASPECT_RATIO_GROUPING = False is not supported yet")