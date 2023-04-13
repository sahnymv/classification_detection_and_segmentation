# References:
    # https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit#slide=id.g15092aa245_0_319
    # https://wolfy.tistory.com/259

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import xml.etree.ElementTree as et
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import extcolors

from process_images import (
    _to_array,
    _to_pil,
    load_image,
    _get_width_and_height,
    show_image
)

voc_classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def parse_voc2012_xml_file(xml_path, voc_classes=voc_classes):
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()

    img = load_image(Path(xml_path).parent.parent/"JPEGImages"/xroot.find("filename").text)
    
    bboxes = torch.Tensor([
        [int(coord.text) for coord in obj.find("bndbox")] + [voc_classes.index(obj.find("name").text)]
        for obj
        in xroot
        if obj.tag == "object"
    ])
    return img, bboxes


def draw_bboxes(img, bboxes):
    img_pil = _to_pil(img)
    draw = ImageDraw.Draw(img_pil)
    for x1, y1, x2, y2 in bboxes[["x1", "y1", "x2", "y2"]].values:
        draw.rectangle(
            xy=(x1, y1, x2, y2),
            outline=(0, 255, 0),
            fill=None,
            width=2
        )
    return _to_array(img_pil)


def generate_ground_truth(xml_path, img_size=448, n_cells=7):
    xml_path="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations/2007_000032.xml"
    img, bboxes = parse_voc2012_xml_file(xml_path)
    w, h = _get_width_and_height(img)
    if w >= h:
        resize_ratio = img_size / w
        # new_w = round(w * resize_ratio)
        new_h = round(h * resize_ratio)
        pad = (img_size - new_h) // 2

        # bboxes[["x1", "y1", "x2", "y2"]] *= resize_ratio
        # bboxes[["y1", "y2"]] += pad
        bboxes["x1"] = bboxes["x1"].apply(lambda x: round(x * resize_ratio))
        bboxes["x2"] = bboxes["x2"].apply(lambda x: round(x * resize_ratio))
        bboxes["y1"] = bboxes["y1"].apply(lambda x: round(x * resize_ratio + pad))
        bboxes["y2"] = bboxes["y2"].apply(lambda x: round(x * resize_ratio + pad))

    copied_bboxes = bboxes.copy()

    copied_bboxes["x"] = copied_bboxes.apply(
        lambda x: (((x["x1"] + x["x2"]) / 2) % (img_size // n_cells)) / (img_size // n_cells),
        axis=1
    )
    copied_bboxes["y"] = copied_bboxes.apply(
        lambda x: (((x["y1"] + x["y2"]) / 2) % (img_size // n_cells)) / (img_size // n_cells),
        axis=1
    )
    copied_bboxes["w"] = copied_bboxes.apply(lambda x: (x["x2"] - x["x1"]) / img_size, axis=1)
    copied_bboxes["h"] = copied_bboxes.apply(lambda x: (x["y2"] - x["y1"]) / img_size, axis=1)

    copied_bboxes["c"] = 1
    copied_bboxes["x_grid"] = copied_bboxes.apply(
        lambda x: int((x["x1"] + x["x2"]) / 2 / (img_size / n_cells)), axis=1
    )
    copied_bboxes["y_grid"] = copied_bboxes.apply(
        lambda x: int((x["y1"] + x["y2"]) / 2 / (img_size / n_cells)), axis=1
    )
    return copied_bboxes[["x_grid", "y_grid", "x", "y", "w", "h", "c", "obj"]]

    # gt = torch.zeros((30, n_cells, n_cells), dtype=torch.float64)
    # for tup in copied_bboxes.itertuples():
    #     _, _, _, _, _, obj, x, y, w, h, c, x_grid, y_grid = tup

    #     if torch.equal(gt[0: 5, y_grid, x_grid], torch.Tensor([0, 0, 0, 0, 0])):
    #         gt[0, y_grid, x_grid] = x
    #         gt[1, y_grid, x_grid] = y
    #         gt[2, y_grid, x_grid] = w ** 0.5
    #         gt[3, y_grid, x_grid] = h ** 0.5
    #         gt[4, y_grid, x_grid] = c
    #         gt[9 + obj, y_grid, x_grid] = 1
    # return gt
        


# def exract_all_colors_from_segmentation_map(seg_map):
#     h, w, _ = seg_map.shape
#     colors = [
#         color
#         for color, _
#         in extcolors.extract_from_image(img=_to_pil(seg_map), tolerance=0, limit=w * h)[0]
#     ]
#     return colors


# def get_minimum_area_mask_bounding_rectangle(mask):
#     bool = (mask != 0)
#     nonzero_x = np.where(bool.any(axis=0))[0]
#     nonzero_y = np.where(bool.any(axis=1))[0]
#     if len(nonzero_x) != 0 and len(nonzero_y) != 0:
#         x1 = nonzero_x[0]
#         x2 = nonzero_x[-1]
#         y1 = nonzero_y[0]
#         y2 = nonzero_y[-1]
#     return x1, y1, x2, y2


# def get_bboxes_from_segmentation_map(seg_map):
#     colors = exract_all_colors_from_segmentation_map(seg_map)
#     ltrbs = [
#         get_minimum_area_mask_bounding_rectangle(
#             np.all(seg_map == np.array(color), axis=2)
#         )
#         for color
#         in colors
#         if color not in [(0, 0, 0), (224, 224, 192)]
#     ]
#     bboxes = pd.DataFrame(ltrbs, columns=("x1", "y1", "x2", "y2"))
#     return bboxes


class Transform(object):
        def __call__(self, image):
            h, w = image.size
            transform = T.Compose(
                [
                    T.ToTensor(),
                    T.CenterCrop(max(h, w)),
                    T.Resize(448, antialias=True),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )
            x = transform(image)
            return x


def pad_to_bboxes(batch):
    max_n_bboxes = max([bboxes.shape[0] for _, bboxes in batch])
    image_list, bboxes_list, = list(), list()
    for image, bboxes in batch:
        image_list.append(image)
        bboxes_list.append(F.pad(bboxes, pad=(0, 0, 0, max_n_bboxes - bboxes.shape[0])))
    return torch.stack(image_list), torch.stack(bboxes_list)


class VOC2012Dataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(list(Path(self.root).glob("*.xml")))

    def __getitem__(self, idx):
        xml_path = list(Path(self.root).glob("*.xml"))[idx]
        img, bboxes = parse_voc2012_xml_file(xml_path)
        image = _to_pil(img)
        if self.transform is not None:
            image = self.transform(image)
        return image, bboxes


if __name__ == "__main__":
    transform = Transform()
    ds = VOC2012Dataset(root="/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations", transform=transform)
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True, drop_last=True, collate_fn=pad_to_bboxes)
    for batch, (image, bboxes) in enumerate(dl, start=1):
        image.shape, bboxes.shape
        bboxes[0]

        darknet = Darknet()
        yolo = YOLO(darknet=darknet, n_classes=20)
        pred = yolo(image)


# bboxes = np.array([
#     [int(coord.text) for coord in obj.find("bndbox")] + [voc_classes.index(obj.find("name").text)]
#     for obj
#     in xroot
#     if obj.tag == "object"
# ])
# bboxs = bboxes[:, :4]
# labels = bboxes[:, 4]
