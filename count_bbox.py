import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
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
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import Visualizer as Vis
from Visualizer import *
from detectron2.data import MetadataCatalog, DatasetCatalog
import importlib
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from matplotlib.pyplot import imshow
import detectron2
from detectron2 import engine
DefaultPredictor = engine.DefaultPredictor
import threading


parser = argparse.ArgumentParser()
parser.add_argument('--bucket', type=str)
parser.add_argument('--sample_size', type=int)
parser.add_argument('--num_threads', type=int, default=4)
parser.add_argument('--threshold', type=str)
args = parser.parse_args()

def count_bbox(folder, im_names, result, received_im = None):
    for i in range(len(im_names)):
        if i % 20 == 19: print("{} bucket, threshold={}, sample_size={}, finish {}/{}".format(args.bucket, args.threshold, args.sample_size, i, len(im_names)))
        im_name = im_names[i]
        if received_im is not None: received_im.append(im_name)
        im = PILImage.open(os.path.join(folder, im_name)).convert("RGB")    
        im = np.array(im)
        outputs = predictor(im)
        result.append(len(outputs['instances'].scores))
    print("return")


if __name__ == "__main__":
    threshold = args.threshold
    cfg = get_cfg()
    config = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file("/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(threshold)  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = "/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl"
    predictor = DefaultPredictor(cfg)

    # We can use `Visualizer` to draw the predictions on the image.
    bucket = args.bucket.lower()

    all_ims = "/data/yingshac/MMMHQA/gold_test/" if bucket=='gold' else "/data/yingshac/MMMHQA/distractors/"
    sample_size = args.sample_size
    num_threads = args.num_threads
    im_names = random.sample(os.listdir(all_ims), sample_size)
    checkpoints = []
    for i in range(num_threads):
        checkpoints.append((i*sample_size//num_threads, (i+1)*sample_size//num_threads))
    results = dict((i, []) for i in range(num_threads))
    received_ims = dict((i, []) for i in range(num_threads))
    threads = []
    for i in range(num_threads):
        c = checkpoints[i]
        t = threading.Thread(target=count_bbox, args=(all_ims, im_names[c[0]:c[1]], results[i], received_ims[i]))
        t.start()
        threads.append(t)
        #count_bbox(im_names[c[0]:c[1]], results[i])
    for t in threads:
        t.join()

    assert len(set([i for v in received_ims.values() for i in v])) == sample_size
    #print(received_ims.values())
    data = [n for v in results.values() for n in v]
    n, bins, patches=plt.hist(data, bins = max(data)-min(data))
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Histogram")
    plt.title('{}_{}_{}, avg_num_bboxes = {}'.format(bucket, sample_size, threshold, np.mean(data)))
    plt.savefig('{}_{}_{}.jpg'.format(bucket, sample_size, threshold))