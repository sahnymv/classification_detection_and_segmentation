# References
    # https://github.com/motokimura/yolo_v1_pytorch/blob/master/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import cv2
import numpy as np
from scipy.sparse import coo_matrix


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


def get_whether_object_appear_in_each_cell(bboxes, n_grids=7):
    bboxes["x"] = bboxes.apply(
        lambda x: (x["xmin"] + x["xmax"]) // 2, axis=1
    )
    bboxes["y"] = bboxes.apply(
        lambda x: (x["ymin"] + x["ymax"]) // 2, axis=1
    )
    bboxes["x_grid"] = bboxes["x"].apply(
        lambda x: int(x / (w / n_grids))
    )
    bboxes["y_grid"] = bboxes["y"].apply(
        lambda x: int(x / (h / n_grids))
    )

    rows = bboxes["y_grid"].tolist()
    cols = bboxes["x_grid"].tolist()
    vals = [True] * len(rows)
    obj_i = torch.from_numpy(
        coo_matrix((vals, (rows, cols)), shape=(n_grids, n_grids)).toarray()
    )
    return obj_i


darknet = Darknet()
yolo = YOLO(darknet=darknet, n_classes=20)

new_w = 448
new_h = 448
h, w, _ = img.shape
if w >= h:
    pad_w = 0
    pad_h = (w - h) // 2
else:
    pad_w = (h - w) // 2
    pad_h = 0
pad_w, pad_h
tranform = T.Compose(
    [
        T.ToTensor(),
        T.Pad(padding=(pad_w, pad_h)),
        T.Resize(448),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)
image = tranform(img).unsqueeze(0)

bboxes = parse_voc2012_xml_file(annot_dir/f"""{filename}.xml""")
obj_i = get_whether_object_appear_in_each_cell(bboxes)
dr = draw_bboxes(img, bboxes)
show_image(dr)



n_classes = 20
obj_i_batch = torch.stack([obj_i] * b, axis=0).unsqueeze(1)

prob_loss = output[:, -20:, ...] * obj_i_batch





n_grids = 7
h, w, _ = img.shape



# bboxes["xmin"] += pad_w
# bboxes["xmax"] += pad_w
# bboxes["ymin"] += pad_h
# bboxes["ymax"] += pad_h
temp = convert_tensor_to_array(image)
temp.shape


