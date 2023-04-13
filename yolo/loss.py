# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

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
from parepare_voc2012 import parse_voc2012_xml_file

np.set_printoptions(precision=3, suppress=True)


def resize_image(img, w, h):
    resized_img = cv2.resize(src=img, dsize=(w, h))
    return resized_img


def draw_grids_and_bounding_boxes(img, bboxes, img_size=448, n_grids=7):
    copied_img = img.copy()
    for i in range(1, n_grids):
        val = img_size // n_grids * i
        cv2.line(img=copied_img, pt1=(val, 0), pt2=(val, img_size), color=(255, 0, 0), thickness=1)
        cv2.line(img=copied_img, pt1=(0, val), pt2=(img_size, val), color=(255, 0, 0), thickness=1)

    for tup in bboxes[["x1", "y1", "x2", "y2"]].itertuples():
        _, x1, y1, x2, y2 = tup

        cv2.rectangle(img=copied_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        cv2.line(img=copied_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1)
        cv2.line(img=copied_img, pt1=(x1, y2), pt2=(x2, y1), color=(0, 255, 0), thickness=1)
    return copied_img


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def get_whether_each_predictor_responsible_for_prediction(pred, n_cells=7, n_bboxes=2, n_classes=20):
    # is_resp = torch.stack([gt[:, 4, ...], gt[:, 9, ...]], dim=1)
    # return is_resp

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
    def __init__(self, lamb_coord=5, lamb_noobj = 0.5):
        super().__init__()

        self.lambd_coord = lamb_coord
        self.lambd_noobj = lamb_noobj
    
    def forward(self, gt, pred):
        mse = F.mse_loss(gt, pred, reduction="none")

        is_resp = get_whether_each_predictor_responsible_for_prediction(gt) # $1^{obj}_{ij}$
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
    bboxes = parse_voc2012_xml_file("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations/2007_000032.xml")
    bboxes = normalize_bounding_boxes_coordinats(bboxes)
    gt = get_ground_truth(bboxes)
    gt = gt[None, ...]

    img = load_image("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg")
    h, w, _ = img.shape
    
    transform = Transform()
    ds = VOC2012Dataset(root="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/JPEGImages", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, drop_last=True)
    for batch, image in enumerate(dl, start=1):
        image.shape

        darknet = Darknet()
        yolo = YOLO(darknet=darknet, n_classes=20)
        pred = yolo(image)