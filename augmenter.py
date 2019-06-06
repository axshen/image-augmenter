import cv2
import random
import numpy as np

import functions
from tools import bounding_box, image_points


class Augmenter:
    def __init__(self, randomised, format):
        assert (format == "bbox") | (
            format == "points"), "Incorrect annotation format provided ('bbox' or 'points')"

        self.randomised = randomised
        self.format = format

    def generate_images(self, n, *args):
        """
        Utilises sequence to generate n images.
        """

    def sequence(self, *args):
        """
        Generate sequence of augmentation functions for image pipeline.
        """
        assert len(args) > 0, "No arguments passed into sequence."

    def rotate(self, img, annots, angle):
        assert angle < 30, "Max angle for image rotation too large."

        if self.randomised:
            angle = round((random.random() - 0.5) * 2 * angle)

        out_img, out_annots = self._task(img, annots, functions.rotate, angle)
        return out_img, out_annots

    # TODO: change translate so doesn't doesn't always cut from du, dv (top left)
    def translate(self, img, annots, du, dv):
        if self.randomised:
            du = random.randint(0, du)
            dv = random.randint(0, dv)

        out_img, out_annots = self._task(
            img, annots, functions.translate, du, dv)
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

    def gaussian_blur(self, img, annots, sigma):
        if self.randomised:
            radius = random.randint(0, int(sigma / 2)) * 2 + 1

        blurred_image = cv2.GaussianBlur(img, (sigma, sigma), 0)

        return blurred_image, annots

    def random_noise(self, img, annots, A):
        if self.randomised:
            amplitude = random.random() * A

        empty_noise = np.empty(img.shape, np.uint8)
        noise = cv2.randn(empty_noise, (0), (A))

        if (random.random() > 0.5):
            out_img = img + noise
        else:
            out_img = img - noise

        out_img = np.abs(out_img)

        return out_img, annots

    def _task(self, img, annots, func, *args):
        """
        Implement single augmentation function. Changes data format
        if necessary and forces generated points to be within.
        """
        if (self.format == "bbox"):
            annot_points = bounding_box.to_corner_points(annots)
            out_img, out_annots = func(img, annot_points, *args)
            out_annots = image_points.to_bounding_box(out_annots)

            # repeat if no change
            within = bounding_box.is_within_image(out_annots, out_img)
            if not within:
                out_img, out_annots = func(img, annots, *args)

                if not self.randomised:
                    raise Exception(
                        'Input augmentation argument always generates points outside cropped image.')

        elif (self.format == "points"):
            annots_homog = image_points.homogenous_coordinates(annots)
            out_img, out_annots = func(img, annots_homog, *args)

            # repeat if no change
            within = image_points.are_within_image(out_annots, out_img)
            if not within:
                out_img, out_annots = out_img, out_annots = func(
                    img, annots_homog, *args)

                if not self.randomised:
                    raise Exception(
                        'Input augmentation argument always generates points outside cropped image.')

        return out_img, out_annots
