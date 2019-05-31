import cv2
import numpy as np


def flip(img, points, axis):
    assert (axis == 0) | (
        axis == 1), "Incorrect axis value provided. 0: vertical or 1: horizontal)"
    out_img = cv2.flip(img, axis)
    out_annots = points
    h, w, _ = img.shape

    # horizontal flip
    if (axis == 1):
        out_annots[:, 0] = w - points[:, 0]

    # vertical flip
    elif (axis == 0):
        out_annots[:, 1] = h - points[:, 1]

    return out_img, out_annots.astype(int)
