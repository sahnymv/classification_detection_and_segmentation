from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import xml.etree.ElementTree as et
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import extcolors

voc_classes = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"
]


def load_image(img_path):
    img_path = str(img_path)
    img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    return img


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _to_array(img):
    img = np.array(img)
    return img


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def parse_voc2012_xml_file(xml_path):
    xtree = et.parse(xml_path)
    xroot = xtree.getroot()
    bboxes = pd.DataFrame(
        [
            [int(coord.text) for coord in label.find("bndbox")] + [voc_classes.index(label.find("name").text)]
            for label
            in xroot
            if label.tag == "object"
        ],
        columns=("x1", "y1", "x2", "y2", "label")
    )
    # bboxes["label"] = bboxes["label"].apply(lambda x: voc_classes.index(x))
    return bboxes


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


def exract_all_colors_from_segmentation_map(seg_map):
    h, w, _ = seg_map.shape
    colors = [
        color
        for color, _
        in extcolors.extract_from_image(img=_to_pil(seg_map), tolerance=0, limit=w * h)[0]
    ]
    return colors


def get_minimum_area_mask_bounding_rectangle(mask):
    bool = (mask != 0)
    nonzero_x = np.where(bool.any(axis=0))[0]
    nonzero_y = np.where(bool.any(axis=1))[0]
    if len(nonzero_x) != 0 and len(nonzero_y) != 0:
        x1 = nonzero_x[0]
        x2 = nonzero_x[-1]
        y1 = nonzero_y[0]
        y2 = nonzero_y[-1]
    return x1, y1, x2, y2


def get_bboxes_from_segmentation_map(seg_map):
    colors = exract_all_colors_from_segmentation_map(seg_map)
    ltrbs = [
        get_minimum_area_mask_bounding_rectangle(
            np.all(seg_map == np.array(color), axis=2)
        )
        for color
        in colors
        if color not in [(0, 0, 0), (224, 224, 192)]
    ]
    bboxes = pd.DataFrame(ltrbs, columns=("x1", "y1", "x2", "y2"))
    return bboxes


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


class VOC2012Dataset(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(list(Path(self.root).glob("**/*")))

    def __getitem__(self, idx):
        img_path = list(Path(self.root).glob("**/*"))[idx]
        image = Image.open(img_path)
        
        if self.transform is not None:
            image = self.transform(image)
        return image


# filenames = [line.strip() for line in open("/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/ImageSets/Segmentation/trainval.txt").readlines()]

# img_dir = "/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/JPEGImages"
# seg_map_dir = "/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/SegmentationObject"
# annot_dir = "/Users/jongbeomkim/Downloads/VOCdevkit/VOC2012/Annotations"

# img_dir = Path(img_dir)
# seg_map_dir = Path(seg_map_dir)
# annot_dir = Path(annot_dir)


# for filename in filenames:
#     img = load_image(img_dir/f"""{filename}.jpg""")
#     bboxes = parse_voc2012_xml_file(annot_dir/f"""{filename}.xml""")
#     dr = draw_bboxes(img, bboxes)
#     show_image(dr)
#     # seg_map = load_image(seg_map_dir/f"""{filename}.png""")

#     bboxes

