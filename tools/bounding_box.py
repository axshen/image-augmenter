import numpy as np


class bounding_box():
    # bbox: np.array([xmin, ymin, xmax, ymax])

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
