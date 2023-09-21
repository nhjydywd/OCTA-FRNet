import cv2
import os
import numpy as np
import random
import json
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
from tqdm import tqdm
from typing import *
import shutil


from models.common import *
from utils import *
from loss import *



import datetime
t = str(datetime.datetime.now()).split(".")[0]
t = t.replace(":",".")
log_root = f"log/{t}"
writer = SummaryWriter(log_dir=log_root)

path_ROSSA = "dataset/ROSSA"
path_OCTA500_6M = "dataset/OCTA500_6M"
path_OCTA500_3M = "dataset/OCTA500_3M"
def prepareDatasets():
    all_datasets = {}
    all_datasets['OCTA500_6M'] = {
        "train":SegmentationDataset(os.path.join(path_OCTA500_6M,"train"), ),
        "val":SegmentationDataset(os.path.join(path_OCTA500_6M, "val")),
        "test":SegmentationDataset(os.path.join(path_OCTA500_6M,"test"))
    }
    all_datasets['OCTA500_3M'] = {
        "train":SegmentationDataset(os.path.join(path_OCTA500_3M,"train"), ),
        "val":SegmentationDataset(os.path.join(path_OCTA500_3M, "val")),
        "test":SegmentationDataset(os.path.join(path_OCTA500_3M,"test"))
    }
    all_datasets['ROSSA'] = {
        "train":SegmentationDataset([os.path.join(path_ROSSA, x) for x in ["train_manual", "train_sam"]], ),
        "val":SegmentationDataset(os.path.join(path_ROSSA, "val")),
        "test":SegmentationDataset(os.path.join(path_ROSSA,"test"))
    }
    # all_datasets['ROSSA_M100'] = {
    #     "train":SegmentationDataset(os.path.join(path_ROSSA, "train_manual") ),
    #     "val":SegmentationDataset(os.path.join(path_ROSSA, "val")),
    #     "test":SegmentationDataset(os.path.join(path_ROSSA,"test"))
    # }
    # all_datasets['ROSSA_S100'] = {
    #     "train":SegmentationDataset(os.path.join(path_ROSSA, "train_sam"), end=100/618 ),
    #     "val":SegmentationDataset(os.path.join(path_ROSSA, "val")),
    #     "test":SegmentationDataset(os.path.join(path_ROSSA,"test"))
    # }
    # all_datasets['ROSSA_S200'] = {
    #     "train":SegmentationDataset(os.path.join(path_ROSSA, "train_sam"), end=200/618 ),
    #     "val":SegmentationDataset(os.path.join(path_ROSSA, "val")),
    #     "test":SegmentationDataset(os.path.join(path_ROSSA,"test"))
    # }
    # all_datasets['ROSSA_S618'] = {
    #     "train":SegmentationDataset(os.path.join(path_ROSSA, "train_sam") ),
    #     "val":SegmentationDataset(os.path.join(path_ROSSA, "val")),
    #     "test":SegmentationDataset(os.path.join(path_ROSSA,"test"))
    # }
    return all_datasets

class SegmentationDataset(Dataset):
    def __init__(self, ls_path_dataset, start=0, end=1) -> None:
        super().__init__()


        if not isinstance(ls_path_dataset, list):
            ls_path_dataset = [ls_path_dataset]

        self.ls_item = []
        for path_dataset in ls_path_dataset:
            path_dir_image = os.path.join(path_dataset, "image")
            path_dir_label = os.path.join(path_dataset, "label")


            ls_file = os.listdir(path_dir_image)
            
            for name in ls_file:
                path_image = os.path.join(path_dir_image, name)
                path_label = os.path.join(path_dir_label, name)
                assert os.path.exists(path_image)
                assert os.path.exists(path_label)
                self.ls_item.append({
                    "name":name,
                    "path_image":path_image,
                    "path_label":path_label,
                })  
        
        random.seed(0)
        random.shuffle(self.ls_item)
        start = int(start * len(self.ls_item))
        end = int(end * len(self.ls_item))
        self.ls_item = self.ls_item[start:end]
          

    def __len__(self):
        return len(self.ls_item)

    def __getitem__(self, index):
        index = index % len(self)
        item = self.ls_item[index]

        name = item['name']
        path_image = item['path_image']
        path_label = item['path_label']

        image = cv2.imread(path_image,cv2.IMREAD_GRAYSCALE).astype("float32")
        label = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE).astype("float32")

        image /= 255
        label /= 255

        # 填充到32的倍数
        pad_x = (image.shape[1] // 32 + 1) * 32 - image.shape[1]
        pad_x %= 32
        pad_y = (image.shape[0] // 32 + 1) * 32 - image.shape[0]
        pad_y %= 32
        image = cv2.copyMakeBorder(image, pad_y//2, pad_y//2, pad_x//2, pad_x//2, cv2.BORDER_CONSTANT, value=0)
        label = cv2.copyMakeBorder(label, pad_y//2, pad_y//2, pad_x//2, pad_x//2, cv2.BORDER_CONSTANT, value=0)

        image = image.reshape((1, image.shape[0], image.shape[1]))
        label = label.reshape((1,label.shape[0], label.shape[1]))


       
        return name, image, label

