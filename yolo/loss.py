# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py
    # https://www.harrysprojects.com/articles/yolov1.html

# For all predicted boxes that are not matched with a ground truth box, it is minimising the objectness confidence, but ignoring the box coordinates and class probabilities.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image


from yolo import Darknet, YOLO
from parepare_voc2012 import Transform, VOC2012Dataset
from process_images import (
    resize_image
)

np.set_printoptions(precision=3, suppress=True)


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def get_whether_each_predictor_is_responsible(pred, n_cells=7, n_bboxes=2, n_classes=20):
    b, _, _, _ = pred.shape
    confs = pred[:, (4, 9), ...]
    argmax = torch.argmax(confs, dim=1)
    onehot = F.one_hot(argmax, num_classes=2).permute(0, 3, 1, 2)
    is_resp = torch.concat(
        [onehot[:, k: k + 1, ...].repeat(1, 5, 1, 1) for k in range(n_bboxes)] +\
            [torch.ones((b, n_classes, n_cells, n_cells))],
        dim=1
    )
    return is_resp


def get_whether_object_appear_in_each_cell(is_resp):
    appears = torch.max(is_resp, dim=1)[0]
    return appears


class Yolov1Loss(nn.Module):
    def __init__(self, lamb_coord=5, lamb_noobj=0.5):
        super().__init__()

        self.lambd_coord = lamb_coord
        self.lambd_noobj = lamb_noobj
    
    def forward(self, gt, pred):
        gt.shape
        gt[0, 4, ...]
        # gt[:, 4, ...] # $1^{obj}_{i, 0}$
        # gt[:, 9, ...] # $1^{obj}_{i, 1}$
        gt[:, 0, ...] * gt[:, 4, ...]


        mse = F.mse_loss(gt, pred, reduction="none")

        is_resp = get_whether_each_predictor_is_responsible(gt) # $1^{obj}_{ij}$
        is_resp.shape
        mse[:, : 4, ...] *= is_resp[:, 0, ...] * self.lamb_coord
        mse[:, 5: 9, ...] *= is_resp[:, 1, ...] * self.lamb_coord

        mse[:, 4, ...] *= (1 - gt[:, 4, ...]) * self.lamb_noobj + gt[:, 4, ...] * self.lamb_coord
        mse[:, 9, ...] *= (1 - gt[:, 9, ...]) * self.lamb_noobj + gt[:, 9, ...] * self.lamb_coord

        appears = get_whether_object_appear_in_each_cell # $1^{obj}_{i}$
        mse[:, 10:, ...] *= appears

        # mse[:, : 4, ...].sum() + mse[:, 5: 9, ...].sum()
        # mse[:, 4, ...].sum() + mse[:, 9, ...].sum()
        # mse[:, 10:, ...].sum()
        # mse[:, 4, ...]
        # mse[:, : 4, ...].sum(dim=1)
        return mse.sum().item()


    # pred = torch.rand((8, 30, 7, 7))

    # greater_conf_indices = torch.argmax(pred[:, (4, 9), ...], dim=1)
    # greater_conf_indices * 5
    # pred[:, (0, 5), ...].shap
    # # torch.take_along_dim(pred[:, (0, 5), ...], indices=greater_conf_indices, dim=1).shape
    # torch.stack(
    #     [torch.take(pred[:, (idx, idx + 5), ...], index=greater_conf_indices) for idx in range(5)], dim=1
    # )


if __name__ == "__main__":
    darknet = Darknet()
    yolo = YOLO(darknet=darknet, n_classes=20)

    criterion = Yolov1Loss()

    transform = Transform()
    ds = VOC2012Dataset(root="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, drop_last=True)
    for batch, (image, gt) in enumerate(dl, start=1):
        pred = yolo(image)
        