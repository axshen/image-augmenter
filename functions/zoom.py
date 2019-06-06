from tools import image_points

import cv2
import numpy as np
import sys

sys.path.append("..")


def zoom(image, points, fx, fy):
    """
    Zoom image by factor fx in the x-axis and fy in the y axis. Rescale 
    image and correct point coordinates to fit reshaped image.
    """
    h, w, _ = image.shape
    out_img = cv2.resize(image, None, fx=fx, fy=fy,
                         interpolation=cv2.INTER_CUBIC)

    # image correction
    aug_h, aug_w, _ = out_img.shape
    h_low = int((aug_h - h) / 2)
    h_up = h + h_low
    w_low = int((aug_w - w) / 2)
    w_up = w + w_low
    out_img = out_img[h_low:h_up, w_low:w_up, :]

    # correct annotations
    out_annots = points
    out_annots[:, 0] = points[:, 0] * fx - int(w_low)
    out_annots[:, 1] = points[:, 1] * fy - int(h_low)
    out_annots = out_annots[:, :-1]

    out_img, out_annots = image_points.rescale(
        out_annots.astype(int), out_img, image.shape)

    return out_img, out_annots
