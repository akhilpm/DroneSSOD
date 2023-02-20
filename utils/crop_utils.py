import copy
import numpy as np
import torch
from torchvision.transforms import Resize
from detectron2.structures.instances import Instances
from croptrain.data.detection_utils import read_image
from detectron2.structures.boxes import Boxes

def get_dict_from_crops(crops, input_dict, CROPSIZE, inner_crop=False):
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
        if inner_crop:
            crop_dict["two_stage_crop"] = True
            crop_dict["inner_crop_area"] = np.array([x1, y1, x2, y2]).astype(np.int32)
        else:    
            crop_dict['crop_area'] = np.array([x1, y1, x2, y2]).astype(np.int32)
        crop_region = read_image(crop_dict)
        crop_region = torch.as_tensor(np.ascontiguousarray(crop_region.transpose(2, 0, 1)))
        crop_region = transform(crop_region)
        crop_dict["image"] = crop_region
        crop_dict["height"] = (y2-y1)
        crop_dict["width"] = (x2-x1)
        crop_dicts.append(crop_dict)
        crop_scales.append(float(CROPSIZE)/crop_size_min)
    return crop_dicts


def project_boxes_to_image(data_dict, crop_sizes, boxes):
    num_bbox_reg_classes = boxes.shape[1] // 4
    output_height, output_width = data_dict.get("height"), data_dict.get("width")
    new_size = (output_height, output_width)
    scale_x, scale_y = (
        output_width / crop_sizes[1],
        output_height / crop_sizes[0],
    )
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.scale(scale_x, scale_y)
    boxes.clip(new_size)
    boxes = boxes.tensor

    #shift to the proper position of the crop in the image
    if not data_dict["full_image"]:
        if data_dict["two_stage_crop"]:
            x1, y1 = data_dict['inner_crop_area'][0], data_dict['inner_crop_area'][1]
            ref_point = torch.tensor([x1, y1, x1, y1]).to(boxes.device)
            boxes = boxes + ref_point
        x1, y1 = data_dict["crop_area"][0], data_dict["crop_area"][1]
        ref_point = torch.tensor([x1, y1, x1, y1]).to(boxes.device)
        boxes = boxes + ref_point
    boxes = boxes.view(-1, num_bbox_reg_classes * 4) # R x C.4
    return boxes    