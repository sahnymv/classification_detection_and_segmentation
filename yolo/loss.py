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


# def get_whether_each_predictor_responsible_for_prediction(bboxes, img_size=448, n_grids=7):
    # img_size=448
    # n_grids=7

    # copied_bboxes["x"] = copied_bboxes.apply(lambda x: (x["x1"] + x["x2"]) / 2, axis=1)
    # copied_bboxes["y"] = copied_bboxes.apply(lambda x: (x["y1"] + x["y2"]) / 2, axis=1)
    
    # for x_grid, y_grid, x, y, w, h, c in copied_bboxes[["x_grid", "y_grid", "x", "y", "w", "h", "conf"]].values:
    #     if tensor[y_grid, x_grid, 0]:
    #         if is_resp[y_grid, x_grid, 1]:
    #             continue
    #         else:
    #             is_resp[y_grid, x_grid, 1] = True
    #     else:
    #         is_resp[y_grid, x_grid, 0] = True
    # return is_resp


# def get_whether_object_appear_in_each_cell(is_resp):
#     appears = torch.max(is_resp, dim=2)[0]
#     return appears


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

    for _, row in bboxes.iterrows():
        cv2.rectangle(img=copied_img, pt1=(row["x1"], row["y1"]), pt2=(row["x2"], row["y2"]), color=(0, 255, 0), thickness=1)
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

    copied_bboxes["conf"] = 1 
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
    copied_bboxes["x_grid"] = copied_bboxes["x_grid"].astype("int")

    gt = torch.zeros((n_grids, n_grids, 30), dtype=torch.float64)
    empty_tensor = torch.Tensor([0, 0, 0, 0, 0])
    for x_grid, y_grid, x, y, w, h, c, label in copied_bboxes[
        ["x_grid", "y_grid", "x", "y", "w", "h", "conf", "label"]
    ].values:
        if torch.equal(gt[int(y_grid), int(x_grid), : 5], empty_tensor):
            gt[int(y_grid), int(x_grid), 0] = x
            gt[int(y_grid), int(x_grid), 1] = y
            gt[int(y_grid), int(x_grid), 2] = w
            gt[int(y_grid), int(x_grid), 3] = h
            gt[int(y_grid), int(x_grid), 4] = c
            gt[int(y_grid), int(x_grid), 9 + int(label)] = 1
        else:
            if torch.equal(gt[int(y_grid), int(x_grid), 5:], empty_tensor):
                gt[int(y_grid), int(x_grid), 5] = x
                gt[int(y_grid), int(x_grid), 6] = y
                gt[int(y_grid), int(x_grid), 7] = w
                gt[int(y_grid), int(x_grid), 8] = h
                gt[int(y_grid), int(x_grid), 9] = c
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


if __name__ == "__main__":
    darknet = Darknet()
    yolo = YOLO(darknet=darknet, n_classes=20)

    img = load_image("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/JPEGImages/2007_000032.jpg")
    bboxes = parse_voc2012_xml_file("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations/2007_000032.xml")
    bboxes = normalize_bounding_boxes_coordinats(bboxes)
    gt = get_ground_truth(bboxes)
    gt.shape


    # is_resp = get_whether_each_predictor_responsible_for_prediction(bboxes)
    # appears = get_whether_object_appear_in_each_cell(is_resp)

    h, w, _ = img.shape
    if w >= h:
        pad_w = 0
        pad_h = (w - h) // 2
    else:
        pad_w = (h - w) // 2
        pad_h = 0

    tranform = T.Compose(
        [
            T.ToTensor(),
            T.Pad(padding=(pad_w, pad_h)),
            T.Resize(448, antialias=True),
            # T.CenterCrop(448),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )
    image = tranform(img).unsqueeze(0)
    arr = tensor_to_array(image)
    drawn = draw_grids_and_bounding_boxes(img=arr, bboxes=bboxes)
    show_image(drawn)

        
    # show_image(arr)
    output = yolo(image)
    output.shape
