import torch
import torch.nn.functional as F
import numpy as np
from itertools import product
from PIL import Image, ImageDraw, ImageFont
import cv2
import pandas as pd

from loss import get_whether_each_predictor_responsible_for_prediction


def _to_pil(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    return img


def _to_array(img):
    img = np.array(img)
    return img


def _blend_two_images(img1, img2, alpha=0.5):
    img1 = _to_pil(img1)
    img2 = _to_pil(img2)
    img_blended = Image.blend(im1=img1, im2=img2, alpha=alpha)
    return _to_array(img_blended)


def denormalize_array(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    copied_img = img.copy()
    copied_img *= variance
    copied_img += mean
    copied_img *= 255.0
    copied_img = np.clip(a=copied_img, a_min=0, a_max=255).astype("uint8")
    return copied_img


def _tensor_to_array(tensor):
    copied_tensor = tensor.clone().squeeze().permute((1, 2, 0)).detach().cpu().numpy()
    copied_tensor = denormalize_array(copied_tensor)
    return copied_tensor


def get_class_probability_maps(pred):
    rgb_vals = [0, 128, 255]
    colors = tuple(product(rgb_vals, rgb_vals, rgb_vals))
    palette = colors[0: 1] + colors[7:]
    
    argmax = pred[:, 10:, ...].argmax(dim=1)
    np.vectorize(lambda x: palette[x])(argmax)
    class_prob_maps = np.stack(np.vectorize(lambda x: palette[x])(argmax), axis=3).astype("uint8")
    return class_prob_maps


def visualize_class_probability_maps(class_prob_maps, image, idx=0):
    img = _tensor_to_array(image[idx])
    class_prob_map = class_prob_maps[idx]
    resized = cv2.resize(class_prob_map, img.shape[: 2], fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    for k in range(7):
        resized[64 * k - 1: 64 * k + 1, :, :] = (0, 0, 0)
    for k in range(7):
        resized[:, 64 * k - 1: 64 * k + 1, :] = (0, 0, 0)
    blended = _blend_two_images(img1=img, img2=resized, alpha=0.7)
    return blended


def tensor_to_array(image, idx=0, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = image.clone()[idx].permute((1, 2, 0)).detach().cpu().numpy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def draw_bboxes(image, bboxes, idx):
    img = tensor_to_array(image=image, idx=idx)
    canvas = _to_pil(img)
    draw = ImageDraw.Draw(canvas)

    for x, y, w, h, c in bboxes.values:
        draw.rectangle(
            xy=(x - w // 2, y - h // 2, x + w // 2, y + h // 2),
            outline=(0, 0, 0),
            fill=None,
            width=int(c * 5)
        )
    canvas = _to_array(canvas)
    return canvas


def draw_predicted_bboxes(pred, image, idx, img_size=448, n_cells=7):
    is_resp = get_whether_each_predictor_responsible_for_prediction(pred)
    pred = is_resp * pred

    # x, y
    cell_size = img_size // n_cells
    pred[:, 0: 2, ...] *= cell_size
    pred[:, 5: 7, ...] *= cell_size
    pred[:, 0: 2, ...] += np.indices((n_cells, n_cells))[1] * cell_size
    pred[:, 5: 7, ...] += np.indices((n_cells, n_cells))[0] * cell_size

    # w, h
    pred[:, 2: 4, ...] *= img_size
    pred[:, 7: 9, ...] *= img_size

    concated = np.concatenate([pred[idx, : 5, ...].reshape(5, -1), pred[idx, 5: 10, ...].reshape(5, -1)], axis=1)
    bboxes = pd.DataFrame(concated.T, columns=["x", "y", "w", "h", "c"])
    bboxes[["x", "y", "w", "h"]] = bboxes[["x", "y", "w", "h"]].astype("int")

    canvas = draw_bboxes(image=image, bboxes=bboxes, idx=idx)
    return canvas


def show_image(img):
    copied_img = img.copy()
    copied_img = _to_pil(copied_img)
    copied_img.show()


def save_image(img, path):
    _to_pil(img).save(str(path))


if __name__ == "__main__":
    pred = torch.rand((8, 30, 7, 7))
    b, _, _, _ = pred.shape

    class_prob_maps = get_class_probability_maps(pred)
    for idx in range(b):
        vis = visualize_class_probability_maps(class_prob_maps=class_prob_maps, image=image, idx=idx)
        show_image(vis)
        save_image(
            img=vis,
            path=f"""/Users/jongbeomkim/Desktop/workspace/segmentation_and_detection/yolo/class_probability_maps/{idx + 1}.jpg"""
        )

    for idx in range(b):
        dr = draw_predicted_bboxes(pred=pred, image=image, idx=idx)
        show_image(dr)
        save_image(
                img=dr,
                path=f"""/Users/jongbeomkim/Desktop/workspace/segmentation_and_detection/yolo/predicted_bboxes/{idx + 1}.jpg"""
            )