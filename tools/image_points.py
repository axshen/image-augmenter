import cv2
import numpy as np


class image_points():
    def to_bounding_box(points):
        """
        points: np.array([[x1, y1],[x2, y2] ... [xn, yn]])
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

    def homogenous_coordinates(points):
        ones = np.ones((points.shape[0], 1))
        points_homog = np.append(points, ones, axis=1)
        return points_homog

    def are_within_image(points, img):
        h, w, _ = img.shape
        within = True
        for point in points:
            if (0 > point[0]) | (point[0] > w) | (0 > point[1]) | (point[1] > h):
                within = False
        return within
