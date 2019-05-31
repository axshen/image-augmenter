import cv2
import random
import numpy as np

from utils import *


def rotation(img, annots, max_angle):
    """
    Script for rotating the image, cropping out empty parts of image (black caused
    by rotation), correcting bounding box, and re-rotating bounding box to
    new rotated, cropped image (was hard).
    Max angle is magnitude (value can be negative or positive)
    """
    angle = round((random.random() - 0.5) * 2 * max_angle)

    # ----------------------
    # FIRST ROTATION
    h, w, _ = img.shape
    cx, cy = (w / 2), (h / 2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    img_out = cv2.warpAffine(img, M, (w, h))
    annots_out = annots.copy()

    xmin, ymin, xmax, ymax = annots

    # image points
    TL_annot = np.array([xmin, ymin, 1])
    TR_annot = np.array([xmax, ymin, 1])
    BR_annot = np.array([xmax, ymax, 1])
    BL_annot = np.array([xmin, ymax, 1])

    out_TL_annot = np.dot(M, TL_annot.T).astype(int)
    out_TR_annot = np.dot(M, TR_annot.T).astype(int)
    out_BR_annot = np.dot(M, BR_annot.T).astype(int)
    out_BL_annot = np.dot(M, BL_annot.T).astype(int)

    # ----------------------
    # CUTTING ORIGINAL IMAGE
    TL_img = np.array([0, 0, 1]).astype(float)
    TR_img = np.array([w, 0, 1]).astype(float)
    BR_img = np.array([w, h, 1]).astype(float)
    BL_img = np.array([0, h, 1]).astype(float)
    out_TL_img = np.dot(M, TL_img.T).astype(int)
    out_TR_img = np.dot(M, TR_img.T).astype(int)
    out_BR_img = np.dot(M, BR_img.T).astype(int)
    out_BL_img = np.dot(M, BL_img.T).astype(int)
    d_xmin = np.abs(0 - out_BL_img[0])
    d_xmax = np.abs(w - out_TR_img[0])
    d_ymin = np.abs(0 - out_TL_img[1])
    d_ymax = np.abs(h - out_BR_img[1])
    img_out = img_out[d_ymin:(h - d_ymax), d_xmin:(w - d_xmax), :]

    # ----------------------
    # UPDATING ANNOTATION
    out_TL_annot -= np.array([d_xmin, d_ymin])
    out_TR_annot -= np.array([d_xmin, d_ymin])
    out_BR_annot -= np.array([d_xmin, d_ymin])
    out_BL_annot -= np.array([d_xmin, d_ymin])

    # ----------------------
    # ROTATING ANNOTATION
    out_C_annot = np.mean(np.array(
        [out_TL_annot, out_TR_annot, out_BR_annot, out_BL_annot]), axis=0).astype(int)
    M_annots = cv2.getRotationMatrix2D(
        (out_C_annot[0], out_C_annot[1]), -angle, 1.0)

    TL_annot_rot = np.array([out_TL_annot[0], out_TL_annot[1], 1])
    TR_annot_rot = np.array([out_TR_annot[0], out_TR_annot[1], 1])
    BR_annot_rot = np.array([out_BR_annot[0], out_BR_annot[1], 1])
    BL_annot_rot = np.array([out_BL_annot[0], out_BL_annot[1], 1])
    out_TL_annot_rot = np.dot(M_annots, TL_annot_rot.T).astype(int)
    out_TR_annot_rot = np.dot(M_annots, TR_annot_rot.T).astype(int)
    out_BR_annot_rot = np.dot(M_annots, BR_annot_rot.T).astype(int)
    out_BL_annot_rot = np.dot(M_annots, BL_annot_rot.T).astype(int)

    # ----------------------
    # CONDENSE
    points = np.array([out_TL_annot_rot, out_TR_annot_rot,
                       out_BR_annot_rot, out_BL_annot_rot])
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    xmax = np.max(points[:, 0])
    ymax = np.max(points[:, 1])
    annots_out = np.array([xmin, ymin, xmax, ymax])

    # ----------------------
    # SCALE TO FULL SIZE
    aug_h, aug_w, _ = img_out.shape
    img_out = cv2.resize(img_out, (1920, 1080))
    h_scale, w_scale = (1080. / aug_h), (1920. / aug_w)
    annots_out = np.array([w_scale * (annots_out[0]),
                           h_scale * (annots_out[1]),
                           w_scale * (annots_out[2]),
                           h_scale * (annots_out[3])]).astype(int)

    within = box_in_image(img_out, annots_out)
    if not within:
        img_out, annots_out = img, annots

    return img_out, annots_out


def flip(img, annots, axis):

    out_img = cv2.flip(img, axis)
    h, w, _ = img.shape
    xmin, ymin, xmax, ymax = annots

    # vertical flip
    if (axis == 0):
        ymin_ = h - ymax
        ymax_ = h - ymin
        out_annots = np.array([xmin, ymin_, xmax, ymax_]).astype(int)
    # horizontal flip
    elif (axis == 1):
        xmin_ = w - xmax
        xmax_ = w - xmin
        out_annots = np.array([xmin_, ymin, xmax_, ymax]).astype(int)
    else:
        out_annots = np.array([xmin, ymin, xmax, ymax]).astype(int)

    within = box_in_image(out_img, out_annots)
    if not within:
        out_img, out_annots = img, annots

    return out_img, out_annots


def zoom(img, annots, max_factor):

    # zoom in and crop image
    factor = (max_factor - 1.) * random.random() + 1.
    out_img = cv2.resize(img, None, fx=factor, fy=factor,
                         interpolation=cv2.INTER_CUBIC)
    aug_h, aug_w, _ = out_img.shape
    h_low = int((aug_h - 1080) / 2)
    h_up = 1080 + h_low
    w_low = int((aug_w - 1920) / 2)
    w_up = 1920 + w_low
    out_img = out_img[h_low:h_up, w_low:w_up, :]

    # correct annotations
    out_annots = np.array([int(factor * annots[0]) - w_low,
                           int(factor * annots[1]) - h_low,
                           int(factor * annots[2]) - w_low,
                           int(factor * annots[3]) - h_low]).astype(int)

    within = box_in_image(out_img, out_annots)
    if not within:
        out_img, out_annots = img, annots

    return out_img, out_annots


def translation(img, annots, max_u_shift, max_v_shift):
    """
    translate image by specified max shift and crop
    will be used with flip so single direction translation
    """
    w_shift = random.randint(0, max_u_shift)
    h_shift = random.randint(0, max_v_shift)
    M = np.float32([[1, 0, w_shift], [0, 1, h_shift]])
    h, w, _ = img.shape

    out_img = cv2.warpAffine(img, M, (w, h))

    # annotations
    top_left = np.array([annots[0], annots[1], 1]).astype(float)
    bottom_right = np.array([annots[2], annots[3], 1]).astype(float)

    out_TL = np.dot(M, top_left.T).astype(int)
    out_BR = np.dot(M, bottom_right.T).astype(int)
    out_annots = np.concatenate((out_TL, out_BR), axis=None)

    # crop image and resize
    img_TL = np.array([0, 0, 1]).astype(float)
    img_BR = np.array([w, h, 1]).astype(float)
    out_img_TL = np.dot(M, img_TL.T).astype(int)
    out_img_BR = np.dot(M, img_BR.T).astype(int)
    w_crop = np.abs(out_img_TL[0])
    h_crop = np.abs(out_img_TL[1])
    out_img = out_img[h_crop:(h - h_crop), w_crop:(w - w_crop), :]
    aug_h, aug_w, _ = out_img.shape
    out_img = cv2.resize(out_img, (1920, 1080))

    # correct bounding box and scale appropriately
    h_scale, w_scale = (1080. / aug_h), (1920. / aug_w)
    out_annots = np.array([w_scale * (out_annots[0] - w_crop),
                           h_scale * (out_annots[1] - h_crop),
                           w_scale * (out_annots[2] - w_crop),
                           h_scale * (out_annots[3] - h_crop)]).astype(int)

    within = box_in_image(out_img, out_annots)
    if not within:
        out_img, out_annots = img, annots

    return out_img, out_annots


def blur(img, annots, max_radius):
    """
    Blurring image with Gaussian blur (apply to image) - annotations
    remain unchanged.
    """
    max_radius = int(max_radius / 2)
    radius = random.randint(0, max_radius) * 2 + 1
    blurred_image = cv2.GaussianBlur(img, (radius, radius), 0)
    return blurred_image, annots


def random_noise(img, annots, max_amplitude):
    """
    Adding noise to the image (random noise)
    """
    amplitude = random.random() * max_amplitude
    empty_noise = np.empty(img.shape, np.uint8)
    noise = cv2.randn(empty_noise, (0), (amplitude))
    p = random.random()
    if (p > 0.5):
        out_img = img + noise
    else:
        out_img = img - noise
    out_img = np.abs(out_img)
    return out_img, annots
