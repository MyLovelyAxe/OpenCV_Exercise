"""
Creates a Pytorch dataset to load the Pascal VOC datasets
"""

import os
import torch
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
# from YOLOV3.utils import iou_width_height
from utils import iou_width_height

ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):

    def __init__(self,
                 csv_file,
                 img_dir,
                 label_dir,
                 anchors,
                #  image_size=416,
                 S=[13,26,52], # Scales of cell, e.g. S=13, there are 13x13 cells in one output image
                 C=20, # number of classes
                 transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        # put all anchors with 3 scales together, check config.py to see what does anchors look like
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir,self.annotations.iloc[index, 1])
        # [class, x, y, w, h] -> [x, y, w, h, class]
        # Note that:
        # content of a bbox is the ratio of length r.w.t the whole image
        # i.e. image size is wImg, hImg, and 11, 0.5, 0.5, 0.9, 0.8
        # means label 11, center at [wImg*0.5,hImg*0.5], bbox size is [wImg*0.9,hImg*0.8]
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir,self.annotations.iloc[index,0])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=image,
                                           bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # we have 3 scales, and same number at each scale
        # (S,S): number of grid_cell, i.e. (13,13), (26,26), (52,52)
        # 6: [prob, x, y, w, h, class] prob: probability that there is an object
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 6)) for S in self.S]

        for box in bboxes:
            # box[2:4]: width and height
            # calculate iou of particular box and all anchors, i.e. 9 anchors, 3 anchors for 3 scales
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            # which anchor is the best
            anchor_indices = iou_anchors.argsort(descending=True,dim=0)
            x, y, width, height, class_label = box
            # whether has anchor or not for each of 3 scale
            # each scale should have 1 anchor
            has_anchor = [False,False,False]

            for anchor_idx in anchor_indices:
                # there are 9 anchors, i.e. 3 anchors for 3 scales
                # to get which anchor it is
                # e.g. idx=8, 8//3=2, i.e. scale_idx=2, i.e. the last scale
                # this tells us which target we need to take out from list "targets"
                # here we get: which scale, i.e. 0th or 1st or 2th scale
                scale_idx = anchor_idx // self.num_anchors_per_scale
                # here we get: which anchor in the particular scale
                # i.e. 0th or 1st or 3nd anchor for this scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                # get grid_cell size of this scale
                S = self.S[scale_idx]
                # which cell is located by the bbox
                # i.e. x=0.5, S=13, --> int(6.5) = 6, center is in 6th cell along x-axis
                i, j = int(S*y), int(S*x)
                # targets[scale_idx]: which scale of cell, 13x13 or 26x26 or 52x52
                # [...][anchor_on_scale, i, j, 0]: index with order [anchor_on_scale, i, j, 0],
                # i.e. [which_anchor_of_this_scale,which_cell_along_x, which_scale_on_y, prob]
                # so anchor_taken means whether this cell has object, i.e. prob of whether there is object
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                # attention:
                #    because the iou_anchors was already sorted with descending order
                #    this first anchor which is taken for a particular scale
                #    must be the one with largest iou, i.e. most compatible with box (ground truth)
                #    so first "if" is to get the first largest-iou anchor
                #    second "elif" means after we already have a largest-iou anchor, i.e. the most suitable one
                #    if there is another anchor which also has too large iou, we consider it as repeatition
                #    which can also present the same object (ground truth box) but not as optimal as the first one
                #    then discard/ignore it, i.e. set prob=-1

                # make sure
                #   1) the anchor hasn't been taken, (ps: originally prob of each anchor is 0)
                #   2) we haven't already had an anchor on this particular scale for this bbox
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # specific location of center in this cell
                    # x_cell and y_cell are in [0,1]
                    x_cell = S*x-j # e.g. 6.5 - 6 = 0.5, i.e. middle of x-axis in this cell
                    y_cell = S*y-i
                    width_cell = width*S # e.g. S=13, width=0.5, width_cell=6.5, corresponds to length of 6.5 cells along x-axis
                    height_cell = height*S
                    box_coordinates = torch.tensor([x_cell,y_cell,width_cell,height_cell])
                    # set x, y, w, h of target
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    # set label of target
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                # make sure:
                #   1) the anchor hasn't been taken, (ps: originally prob of each anchor is 0)
                #   2) the anchor's iou is not too large, i.e. anchor doesn't superpose largely with box
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore this prediction, if the anchor superposes too much

        return image, tuple(targets)
