from tools import image_points

import cv2
import numpy as np
import sys

sys.path.append("..")


def translate(img, annots, du, dv):
    M = np.float32([[1, 0, du], [0, 1, dv]])
    h, w, _ = img.shape
    out_img = cv2.warpAffine(img, M, (w, h))
    out_annots = np.dot(M, annots.T).T.astype(int)

    out_img = out_img[dv:(h - dv), du:(w - du), :]
    aug_h, aug_w, _ = out_img.shape
    h_scale, w_scale = (float(h) / aug_h), (float(w) / aug_w)

    out_annots[:, 0] = w_scale * (out_annots[:, 0] - du)
    out_annots[:, 1] = h_scale * (out_annots[:, 1] - dv)

    out_img = cv2.resize(out_img, (w, h))

    return out_img, out_annots.astype(int)
