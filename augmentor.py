import cv2
import random
import numpy as np

import functions
from utils import bounding_box, image_points, cv_draw


class Augmentor:
    def __init__(self, randomised, format):
        assert (format == "bbox") | (
            format == "points"), "Incorrect annotation format provided ('bbox' or 'points')"

        self.randomised = randomised
        self.format = format

    def _task(self, img, annots, func, *args):
        """
        Implement single augmentation function with relevant function.
        """
        if (self.format == "bbox"):
            annot_points = bounding_box.to_corner_points(annots)
            out_img, out_annots = func(img, annot_points, *args)
            out_annots = image_points.to_bounding_box(out_annots)
        elif (self.format == "points"):
            out_img, out_annots = func(img, annots, *args)

        # repeat if no change
        within = bounding_box.is_within_image(out_annots, out_img)
        if not within:
            out_img, out_annots = self._task(img, annots, func, *args)

        return out_img, out_annots

    def rotate(self, img, annots, angle):
        assert angle < 30, "Max angle for image rotation too large."

        if self.randomised:
            angle = round((random.random() - 0.5) * 2 * angle)

        out_img, out_annots = self._task(img, annots, functions.rotate, angle)
        return out_img, out_annots

    def flip(self, img, annots, axis):
        out_img, out_annots = self._task(img, annots, functions.flip, axis)
        return out_img, out_annots

    def zoom(self, img, annots, fx, fy):
        if self.randomised:
            fx = (fx - 1.) * random.random() + 1.
            fy = (fy - 1.) * random.random() + 1.

        out_img, out_annots = self._task(img, annots, functions.zoom, fx, fy)
        return out_img, out_annots

    # TODO: change translate so doesn't doesn't always cut from du, dv (top left)
    def translate(self, img, annots, du, dv):
        if self.randomised:
            du = random.randint(0, du)
            dv = random.randint(0, dv)

        out_img, out_annots = self._task(
            img, annots, functions.translate, du, dv)
        return out_img, out_annots

    def gaussian_blur(self, img, annots, radius):
        if self.randomised:
            radius = random.randint(0, int(radius / 2)) * 2 + 1

        blurred_image = cv2.GaussianBlur(img, (radius, radius), 0)

        return blurred_image, annots

    def random_noise(self, img, annots, max_amplitude):
        """
        Adding or subtracting noise to the image (random noise) 
        """
        amplitude = random.random() * max_amplitude
        empty_noise = np.empty(img.shape, np.uint8)
        noise = cv2.randn(empty_noise, (0), (amplitude))
        if (random.random() > 0.5):
            out_img = img + noise
        else:
            out_img = img - noise
        out_img = np.abs(out_img)

        return out_img, annots


def main():
    augmentor = Augmentor(randomised=True, format='bbox')
    # images_dir = "/Users/austin.shen/Dropbox/Formalytics/deepLearning/workspace/ball_detection/data/archive/images/test/"
    images_dir = "/Users/austin.shen/Downloads"
    test_image = cv2.imread(
        images_dir + "6620fba1-411f-472d-a212-0a78a80ab6c2.jpeg")
    annot = np.array([825, 487, 861, 523])

    for _ in range(10):
        write_img = test_image.copy()

        cv_draw.bounding_box(write_img, annot, (0, 0, 255))
        cv2.imshow("image", write_img)
        cv2.waitKey(1)

        # out_img, out_annots = augmentor.rotate(test_image, annot, 20)
        # out_img, out_annots = augmentor.flip(test_image, annot, 1)
        # out_img, out_annots = augmentor.zoom(test_image, annot, 1.5, 1.5)
        # out_img, out_annots = augmentor.translate(test_image, annot, 200, 200)

        cv_draw.bounding_box(out_img, out_annots, (0, 0, 255))
        cv2.imshow("image", out_img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
