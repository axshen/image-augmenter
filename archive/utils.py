import numpy as np
import cv2
import sys
import math

# ------------------------------------------------------------------------------
# DRAWING FUNCTIONS

def draw_bounding_box(img, box, colour):
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colour, 2)

def draw_point(img, point, colour):
    cv2.circle(img, (point[0], point[1]), 5, colour, -1)

def box_in_image(img,box):
    h, w, c = img.shape
    if (w != 1920 or h != 1080):
        print("Box in image error: image shape incorrect.")
        sys.exit()
    xmin, ymin, xmax, ymax = box
    if (xmin < 0 or ymin < 0 or xmax > w or ymax > h):
        within = False
    else:
        within = True
    return within

# ------------------------------------------------------------------------------
# NUMERIC TRANSFORMATIONS

def log_radius(radii):
    """
    taking the log of the radii values and normalising. Printing the max and min values used
    for normalisation from zi = (xi - min(x))/(max(x) - min(x))
    """
    log_radii = list(map(math.log10, radii))
    log_radii = np.array(log_radii)
    max_r = np.max(log_radii)
    min_r = np.min(log_radii)
    print('max & min values for normalisation: %.5f, %.5f' % (max_r, min_r))
    log_radii = (log_radii - min_r) / (max_r - min_r)
    log_radii = 2 * (log_radii - 0.5)
    return log_radii

def inverse_log(radii, min, max):
    """
    Inverse the process for normalising the radius value of the ball (retrieve real value
    of radius - pixel size in image) as float
    """
    radii = radii / 2  + 0.5
    radii = radii * (max - min) + min
    return radii
