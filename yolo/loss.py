# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
import pandas as pd
from yolo import Darknet, YOLO
from parepare_voc2012 import parse_voc2012_xml_file

np.set_printoptions(precision=3, suppress=True)


def get_whether_each_predictor_responsible_for_prediction(gt):
    is_resp = torch.stack([gt[:, 4, ...], gt[:, 9, ...]], dim=1)
    return is_resp


def get_whether_object_appear_in_each_cell(is_resp):
    appears = torch.max(is_resp, dim=1)[0]
    return appears


def denormalize_array(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    copied_img *= variance
    copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def convert_tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


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


def normalize_bounding_boxes_coordinats(bboxes, img_size=448, n_grids=7):
    copied_bboxes = bboxes.copy()

    copied_bboxes["x"] = copied_bboxes.apply(
        lambda x: (((x["x1"] + x["x2"]) / 2) % (img_size // n_grids)) / (img_size // n_grids),
        axis=1
    )
    copied_bboxes["y"] = copied_bboxes.apply(
        lambda x: (((x["y1"] + x["y2"]) / 2) % (img_size // n_grids)) / (img_size // n_grids),
        axis=1
    )
    copied_bboxes["w"] = copied_bboxes.apply(lambda x: (x["x2"] - x["x1"]) / img_size, axis=1)
    copied_bboxes["h"] = copied_bboxes.apply(lambda x: (x["y2"] - x["y1"]) / img_size, axis=1)

    copied_bboxes["c"] = 1 
    return copied_bboxes


def get_ground_truth(bboxes, img_size=448, n_grids=7):
    # img_size=448
    # n_grids=7
    copied_bboxes = bboxes.copy()

    copied_bboxes["x_grid"] = copied_bboxes.apply(
        lambda x: int((x["x1"] + x["x2"]) / 2 / (img_size / n_grids)), axis=1
    )
    copied_bboxes["y_grid"] = copied_bboxes.apply(
        lambda x: int((x["y1"] + x["y2"]) / 2 / (img_size / n_grids)), axis=1
    )

    gt = torch.zeros((30, n_grids, n_grids), dtype=torch.float64)
    for tup in copied_bboxes.itertuples():
        _, _, _, _, _, label, x, y, w, h, c, x_grid, y_grid = tup

        if torch.equal(gt[0: 5, y_grid, x_grid], torch.Tensor([0, 0, 0, 0, 0])):
            gt[0, y_grid, x_grid] = x
            gt[1, y_grid, x_grid] = y
            gt[2, y_grid, x_grid] = w ** 0.5
            gt[3, y_grid, x_grid] = h ** 0.5
            gt[4, y_grid, x_grid] = c
            gt[9 + label, y_grid, x_grid] = 1
        else:
            if torch.equal(gt[5: 10, y_grid, x_grid], torch.Tensor([0, 0, 0, 0, 0])):
                gt[5, y_grid, x_grid] = x
                gt[6, y_grid, x_grid] = y
                gt[7, y_grid, x_grid] = w ** 0.5
                gt[8, y_grid, x_grid] = h ** 0.5
                gt[9, y_grid, x_grid] = c
            else:
                continue
    return gt


def tensor_to_array(image, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = image.clone()[0].permute((1, 2, 0)).detach().cpu().numpy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


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


if __name__ == "__main__":
    bboxes = parse_voc2012_xml_file("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations/2007_000032.xml")
    bboxes = normalize_bounding_boxes_coordinats(bboxes)
    gt = get_ground_truth(bboxes)
    gt = gt[None, ...]

    img = load_image("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg")
    h, w, _ = img.shape
    class Transform(object):
        # def __init__(self):

        def __call__(self, image):
            h, w, _ = image.shape
            transform = T.Compose(
                [
                    T.ToTensor(),
                    # T.Resize(448),
                    # T.CenterCrop(448),
                    T.CenterCrop(max(h, w)),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
            x = transform(image)
            return x
    image = tranform(img).unsqueeze(0)
    arr = tensor_to_array(image)
    show_image(arr)

    darknet = Darknet()
    yolo = YOLO(darknet=darknet, n_classes=20)
    pred = yolo(image)

    drawn = draw_grids_and_bounding_boxes(img=arr, bboxes=bboxes)
    show_image(drawn)

