import numpy as np
from itertools import product
from PIL import Image
import cv2


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
    color_vals = list(range(50, 255 + 1, 255 // 3))
    palette = list(product(color_vals, color_vals, color_vals))[1:]
    
    argmax = pred[:, 10:, ...].argmax(dim=1)
    class_prob_maps = np.stack(np.vectorize(lambda x: palette[x])(argmax), axis=3).astype("uint8")
    return class_prob_maps


def visualize_class_probability_maps(class_prob_maps, image, idx=0):
    img = _tensor_to_array(image[idx])
    class_prob_map = class_prob_maps[idx]
    resized = cv2.resize(class_prob_map, img.shape[: 2], fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    blended = _blend_two_images(img1=img, img2=resized, alpha=0.7)
    return blended


if __name__ == "__main__":
    class_prob_maps = get_class_probability_maps(pred)
    vis = visualize_class_probability_maps(class_prob_maps=class_prob_maps, image=image, idx=3)
    show_image(vis)