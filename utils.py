import numpy as np
import cv2
import sys
import math


class bounding_box():
    # bbox = np.array([xmin, ymin, xmax, ymax])

    def to_corner_points(bbox):
        """
        Convert bounding box to four screen points (u, v)
        """
        xmin, ymin, xmax, ymax = bbox
        TL = np.array([xmin, ymin, 1]).astype(float)
        TR = np.array([xmax, ymin, 1]).astype(float)
        BR = np.array([xmax, ymax, 1]).astype(float)
        BL = np.array([xmin, ymax, 1]).astype(float)
        points = np.array([TL, TR, BR, BL])
        return points

    def is_within_image(bbox, img):
        h, w, _ = img.shape

        xmin, ymin, xmax, ymax = bbox
        if (xmin < 0 or ymin < 0 or xmax > w or ymax > h):
            within = False
        else:
            within = True
        return within


class image_points():
    # points = np.array([[x1, y1],[x2, y2] ... [xn, yn]])

    def to_bounding_box(points):
        """
        Take points describing corners of region and converts to bbox format.
        """
        xmin = np.min(points[:, 0])
        ymin = np.min(points[:, 1])
        xmax = np.max(points[:, 0])
        ymax = np.max(points[:, 1])
        return np.array([xmin, ymin, xmax, ymax])

    def rescale(points, img, shape):
        """
        Rescale image to specified shape and point annotations accordingly
        """
        h, w, _ = shape
        aug_h, aug_w, _ = img.shape
        aug_img = cv2.resize(img, (w, h))
        h_scale, w_scale = (float(h) / aug_h), (float(w) / aug_w)
        aug_annots = points.astype(float)
        aug_annots[:, 0] *= w_scale
        aug_annots[:, 1] *= h_scale
        return aug_img, aug_annots.astype(int)


class cv_draw():
    def bounding_box(img, box, colour):
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colour, 2)

    def draw_point(img, point, colour):
        cv2.circle(img, (point[0], point[1]), 5, colour, -1)
