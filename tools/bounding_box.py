import numpy as np


class bounding_box():
    def to_corner_points(bbox):
        """
        bbox: np.array([xmin, ymin, xmax, ymax])
        Convert bounding box to a numpy array of four screen points (u, v)
        """
        xmin, ymin, xmax, ymax = bbox
        TL = np.array([xmin, ymin, 1]).astype(float)
        TR = np.array([xmax, ymin, 1]).astype(float)
        BR = np.array([xmax, ymax, 1]).astype(float)
        BL = np.array([xmin, ymax, 1]).astype(float)
        points = np.array([TL, TR, BR, BL])
        return points
