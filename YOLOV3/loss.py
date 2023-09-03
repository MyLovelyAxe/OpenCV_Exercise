import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # constants: what are these
        self.lambda_class = 1
        self.lambda_noobj = 10 # value no-object more
        self.lambda_obj = 1
        self.lambda_box = 10 # value box more

    def forward(self, predictions, targets, anchors):
        """
        target: e.g. [num_scale=3,num_cell_x=13,num_cell_y=13,num_params=6]
            6 params: (prob, x, y, w, h, class_label)
        """
        # check whether there is object in all cells, i.e. check if prob==1
        # because some of targets has large iou >= threshold, and we ignore them
        obj = targets[..., 0] == 1
        noobj = targets[..., 0] == 0

        ### No Object Loss
        no_object_loss = self.bce(
            predictions[...,0:1][noobj], targets[..., 0:1][noobj], # prediction[...,0:1]: keep dimension in case of error
        )

        ### Object Loss
        # original shape of anchors: [3,2]
        # i.e. each scale with 3 anchors,
        # each anchor has width and height, i.e. len(dim=1)=2
        # according to formular in paper:
        #   bx = sigmoid(tx) # bx: box center x_coord, tx: output/learned value from network, sigmoid makes tx within (0,1)
        #   by = sigmoid(ty)
        #   bw = pw * exp(tw) # bw: box width, tw: output/learned value from network, pw: encoded knowledge of that particular anchor box, i.e. anchor width
        #   bh = ph * exp(th)
        # "anchors" contains the widths and heights of anchors
        anchors = anchors.reshape(1,3,1,1,2) # in order to be compatible with formula: e.g. bw = pw * exp(tw)
        box_preds = torch.cat([self.sigmoid(predictions[...,1:3]),torch.exp(predictions[...,3:5]*anchors)], dim=-1)
        iou = intersection_over_union(box_preds[obj],targets[obj]).detach()
        object_loss = self.bce(
            predictions[...,0:1][obj], iou*targets[..., 0:1][noobj]
        )

        ### Box Coordinate Loss
        # in order to have better gradient flow, and do something
        predictions[...,1:3] = self.sigmoid(predictions[...,1:3]) # x,y to be within [0,1]
        targets[...,3:5] = torch.log(
            1e-16 + targets[...,3:5] / anchors
        )
        box_loss = self.mse(predictions[...,1:5][obj],targets[...,1:5][obj])

        ### Class Loss
        class_loss = self.entropy(
            predictions[...,5][obj], targets[...,5][obj].long()
        )

        return (
            self.lambda_box * box_loss + 
            self.lambda_obj * object_loss + 
            self.lambda_noobj * no_object_loss + 
            self.lambda_class * class_loss
        )