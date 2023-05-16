# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures.boxes import Boxes
from typing import Dict, List, Optional
from utils.crop_utils import project_boxes_to_image, get_dict_from_crops
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", val_mode=False,
        infer_on_crops: bool = False,
        cfg=None
    ):
        if (not self.training) and (not val_mode):
            if infer_on_crops:
                return self.infer_on_image_and_crops(batched_inputs, cfg)
            else:    
                return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    def get_box_predictions(self, features, proposals):
        features = [features[f] for f in self.roi_heads.box_in_features]
        box_features = self.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.roi_heads.box_head(box_features)
        predictions = self.roi_heads.box_predictor(box_features)
        del box_features
        boxes = self.roi_heads.box_predictor.predict_boxes(predictions, proposals)
        scores = self.roi_heads.box_predictor.predict_probs(predictions, proposals)
        return list(boxes), list(scores)

    def infer_on_image_and_crops(self, inputs, cfg, CROPSIZE=512):
        assert len(inputs)==1, "Only one image per inference is supported!"
        assert not self.training
        crop_class = self.roi_heads.num_classes-1
        image_shapes = [(item.get("height"), item.get("width")) for item in inputs]
        images_original = self.preprocess_image(inputs)
        features_original = self.backbone(images_original.tensor)
        proposals_original, _ = self.proposal_generator(images_original, features_original, None)
        #get detections from full image and project it to original image size
        boxes, scores = self.get_box_predictions(features_original, proposals_original)
        boxes[0] = project_boxes_to_image(inputs[0], images_original.image_sizes[0], boxes[0])
        del images_original,features_original
        if not cfg.CROPTRAIN.USE_CROPS:
            return boxes, scores
        pred_instances, _ = fast_rcnn_inference(boxes, scores, image_shapes, self.roi_heads.box_predictor.test_score_thresh, \
                                self.roi_heads.box_predictor.test_nms_thresh, self.roi_heads.box_predictor.test_topk_per_image)
        crop_class_indices = (pred_instances[0].pred_classes==crop_class)
        crop_boxes = pred_instances[0][crop_class_indices]
        crop_boxes = crop_boxes[crop_boxes.scores>0.85]
        if len(crop_boxes)!=0:
            if inputs[0]["full_image"]:
                crop_dicts = get_dict_from_crops(crop_boxes, inputs[0], CROPSIZE)
            else:
                crop_dicts = get_dict_from_crops(crop_boxes, inputs[0], CROPSIZE=CROPSIZE, inner_crop=True)    
            for i, crop_dict in enumerate(crop_dicts):
                images_crop = self.preprocess_image([crop_dict])
                features_crop = self.backbone(images_crop.tensor)
                proposals_crop, _ = self.proposal_generator(images_crop, features_crop, None)
                #get detections from crop and project it to wrt to original image size
                boxes_crop, scores_crop = self.get_box_predictions(features_crop, proposals_crop)
                boxes_crop = project_boxes_to_image(crop_dict, images_crop.image_sizes[0], boxes_crop[0])
                boxes[0] = torch.cat([boxes[0], boxes_crop], dim=0)
                scores[0] = torch.cat([scores[0], scores_crop[0]], dim=0)
                del images_crop, features_crop    
        if not inputs[0]["full_image"]:
            return boxes, scores
        pred_instances, _ = fast_rcnn_inference(boxes, scores, image_shapes, self.roi_heads.box_predictor.test_score_thresh, \
                                self.roi_heads.box_predictor.test_nms_thresh, self.roi_heads.box_predictor.test_topk_per_image)
        pred_instances = pred_instances[0]
        pred_instances = pred_instances[pred_instances.pred_classes!=crop_class]
        all_outputs = [{"instances": pred_instances}]
        return all_outputs
