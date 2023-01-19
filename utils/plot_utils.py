from detectron2.utils.visualizer import Visualizer
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from detectron2.structures.instances import Instances

def plot_detections(predictions, cluster_boxes, data_dict, metadata, cfg, iter):
    img = np.array(Image.open(data_dict["file_name"]))
    visualizer = Visualizer(img, metadata=metadata)
    #visualizer.draw_instance_predictions(predictions)
    if len(cluster_boxes)==0:
        print("No clusters")
        return
    vis = visualizer.overlay_instances(boxes=cluster_boxes.pred_boxes.tensor.cpu())
    save_path = os.path.join(cfg.OUTPUT_DIR, "detections", str(iter)+'_'+data_dict["file_name"].split('/')[-1])
    vis.save(save_path)

def plot_image(data_dict, cfg, metadata):
    img = np.array(Image.open(data_dict["file_name"]))
    visualizer = Visualizer(img, metadata=metadata)
    vis = visualizer.draw_dataset_dict(data_dict)
    save_path = os.path.join(cfg.OUTPUT_DIR, "input", data_dict["file_name"].split('/')[-1])
    vis.save(save_path)


def plot_pseudo_gt_boxes(data_dict, tag, img=None):
    if img is None:
        img = data_dict["image"] # Image.open(data_dict["file_name"])
    img = img.permute(1, 2, 0)
    plt.axis('off') 
    plt.imshow(img)
    ax = plt.gca()
    gt_instances = data_dict["instances"].to("cpu")
    if len(gt_instances)!=0:
        gt_boxes = gt_instances.gt_boxes.tensor.cpu()
        for bbox in gt_boxes:
            x1, y1 = bbox[0], bbox[1]
            h, w = bbox[3]-bbox[1], bbox[2]-bbox[0]
            rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)
    im_name = os.path.basename(data_dict["file_name"])[:-4]
    plt.savefig(os.path.join('./temp', im_name+"_{}_pgt.jpg".format(tag)), dpi=90, bbox_inches='tight')
    #plt.savefig(os.path.join('./temp', im_name+"_det_Base.jpg"), dpi=90, bbox_inches='tight')
    #plt.show()
    plt.clf()    