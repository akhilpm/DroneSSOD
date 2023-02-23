import torch
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.fcos import FCOS
from detectron2.structures.boxes import Boxes
from typing import Dict, List, Optional, Tuple
from detectron2.utils.events import get_event_storage
from detectron2.structures import Instances
from utils.crop_utils import project_boxes_to_image


@META_ARCH_REGISTRY.register()
class CROP_FCOS(FCOS):

    def forward(self, batched_inputs: List[Dict[str, Tensor]],
        cluster_inputs: Optional[List[Dict[str, torch.Tensor]]] = None,
        infer_on_crops: bool = False,
    ):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            return self.forward_training(images, features, predictions, gt_instances)
        else:
            results = self.forward_inference(images, features, predictions)
            if torch.jit.is_scripting():
                return results

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results            