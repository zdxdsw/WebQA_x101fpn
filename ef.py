import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
import torch
assert torch.__version__.startswith("1.8")
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random
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
from PIL import Image as PILImage
from matplotlib.pyplot import imshow
import detectron2
from detectron2 import engine
DefaultPredictor = engine.DefaultPredictor

cfg = get_cfg()
config = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file("/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = "/home/yingshac/CYS/WebQnA/RegionFeature/detectron-vlp/e2e_faster_rcnn_X-101-64x4d-FPN_2x-vlp-427.pkl"
predictor = DefaultPredictor(cfg)

# We can use `Visualizer` to draw the predictions on the image.
im_name = random.choice(os.listdir("/data/yingshac/MMMHQA/distractors/"))
print(im_name)
im = PILImage.open(os.path.join("/data/yingshac/MMMHQA/distractors/", im_name)).convert("RGB")
im = np.array(im)

outputs = predictor.inference(im)
print(outputs[0].fc1_features.size())
print(outputs[0].cls_features.size())

