import numpy as np
import torch
from torch.utils.data import Dataset
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MyDataset(Dataset):
    def __init__(self, abs_paths, MIN_SIZE_TEST, MAX_SIZE_TEST, USE_CUDA=torch.cuda.is_available()):
        self.abs_paths = abs_paths # list of img absolute paths
        self.USE_CUDA = USE_CUDA

        self.aug = T.ResizeShortestEdge([MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)

    
    def __getitem__(self, index):
        im_name = self.abs_paths[index]
        original_image = utils.read_image(im_name, format="RGB")
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = {"image": image, "height": height, "width": width}
        #if self.USE_CUDA:
            #source = source.cuda()
            #target = target.cuda()
        return inputs, im_name

    def __len__(self):
        return len(self.abs_paths)