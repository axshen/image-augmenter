from utils import bounding_box, image_points

import cv2
import numpy as np
import sys

sys.path.append("..")


def rotate(image, points, angle):
    """
    Function to rotate the image and points by a specified angle.
    """
    h, w, _ = image.shape
    cx, cy = (w / 2), (h / 2)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    out_img = cv2.warpAffine(image, M, (w, h))

    rotated_points = np.dot(M, points.T).T.astype(int)

    # image points
    image_corners = np.array([0, 0, w, h])
    corner_points = bounding_box.to_corner_points(image_corners)
    rotated_corner_points = np.dot(M, corner_points.T).T.astype(int)

    # crop rotated image and update annotations
    out_img, rotated_points = _rotation_crop(
        out_img, rotated_corner_points, rotated_points, image.shape)

    # get inverse rotation matrix
    out_C_annot = np.mean(rotated_points, axis=0).astype(int)
    M_inverse = cv2.getRotationMatrix2D(
        (out_C_annot[0], out_C_annot[1]), -angle, 1.0)

    # inverse rotate annotation points
    ones_column = np.ones((rotated_points.shape[0], 1))
    rotated_points = np.append(rotated_points, ones_column, axis=1)
    out_annots = np.dot(M_inverse, rotated_points.T).T.astype(int)

    # rescale
    out_img, out_annots = image_points.rescale(
        out_annots, out_img, image.shape)

    return out_img, out_annots


def _rotation_crop(rotated_img, rotated_corners, rotated_annots, shape):
    """
    Crop the empty regions of image from rotation. Return cropped image.
    TODO: change constant 4.
    """
    h, w, _ = shape
    d_xmin = np.abs(0 - rotated_corners[3][0])
    d_xmax = np.abs(w - rotated_corners[1][0])
    d_ymin = np.abs(0 - rotated_corners[0][1])
    d_ymax = np.abs(h - rotated_corners[2][1])

    out_img = rotated_img[d_ymin:(h - d_ymax), d_xmin:(w - d_xmax), :]
    out_annots = rotated_annots - np.array([[d_xmin, d_ymin]] * 4)

    return out_img, out_annots