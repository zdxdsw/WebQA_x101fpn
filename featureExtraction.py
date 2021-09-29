import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2, 3"
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import torch
assert torch.__version__.startswith("1.8")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import json, random, argparse
#from google.colab.patches import cv2_imshow
import cv2
from IPython.display import display
from IPython.display import Image as IPyImage
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import Visualizer as Vis
from Visualizer import *
from detectron2.data import MetadataCatalog, DatasetCatalog
from io import BytesIO
from PIL import Image as PILImage
import detectron2
from detectron2 import engine
DefaultPredictor = engine.DefaultPredictor
import threading
import multiprocessing
from pprint import pprint
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import detection_utils as utils
from mydataset import MyDataset
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

def save_RF(im_names_list, outputs_list, save_dir, bucket, append):
    bucket_save_dir = os.path.join(save_dir, bucket)
    
    for im_name, output in zip(im_names_list, outputs_list):
        inst = output['instances']
        im_id = os.path.basename(im_name).replace(".jpg", "")
        region_features = {"image_size": inst.image_size, "num_instances": len(inst),\
                                                                         "pred_boxes": inst.pred_boxes.tensor, "scores": inst.scores, \
                                                                         "pred_classes": inst.pred_classes, \
                                                                          "fc1_features": inst.fc1_features, \
                                                                         "cls_features": inst.cls_features}
        with open(os.path.join(bucket_save_dir, "{}.pkl".format(im_id)), 'wb') as fp:
            pickle.dump(region_features,fp)

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--threshold', type=str, default='0.0')
parser.add_argument('--output_dir', type=str, default="/data/yingshac/MMMHQA/imgFeatures_x_distractors/")
parser.add_argument('--log_path', type=str, default="./log.txt")
parser.add_argument('--start', type=int)
parser.add_argument('--end', type=int)

args = parser.parse_args()
args.bucket = args.input_dir.strip("/").split("/")[-1]
print("input_dir = ", args.input_dir)
print("output_dir = ", args.output_dir)
print("bucket = ", args.bucket)

if __name__ == "__main__":
    threshold = float(args.threshold)
    cfg = get_cfg()
    config = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file("/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # set threshold for this model
    cfg.MODEL.WEIGHTS = "/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl"
    model = build_model(cfg)
    print("Finish building model")
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS) 
    print("Finish loading weights")
    #model = torch.nn.DataParallel(model, device_ids=[0,1,2]).module
    abs_paths = [os.path.join(args.input_dir, im_name) for im_name in os.listdir(args.input_dir)[args.start:args.end]]
    #abs_paths = [os.path.join(args.input_dir, im_name) for im_name in sorted(os.listdir(args.input_dir))[args.start:args.end]]
    print("number of images to load = ", len(abs_paths))
    dataset = MyDataset(abs_paths, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)


    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, collate_fn = lambda x: x)
    print("Finish building dataloader")
    model.eval()
    outputs_list = []
    im_names_list = []
    append = False
    print("start inference")
    with torch.no_grad():
        for batch_idx, item in enumerate(tqdm(dataloader)):
            inputs = [i[0] for i in item]
            im_names = [i[1] for i in item]
            im_names_list.extend(im_names)
            outputs = model.inference_FE(inputs)
            outputs_list.extend(outputs)
            if batch_idx % 30 == 0:
                save_RF(im_names_list, outputs_list, args.output_dir, args.bucket, append)
                append = True
                outputs_list = []
                im_names_list = []
    save_RF(im_names_list, outputs_list, args.output_dir, args.bucket, append)
    
    with open(args.log_path, 'a') as file:
        file.write("Finish {} !!\n".format(args.bucket, args.threshold))
    
    print("Finish !! bucket={}, threshold={}, ".format(args.bucket, args.threshold))
    #assert len(region_features_reload) == len(abs_paths)