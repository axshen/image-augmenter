import cv2
import random
import numpy as np
import sys

import functions
from tools import bounding_box, image_points


class Augmenter:
    def __init__(self, randomised):
        self.randomised = randomised

    def generate_images(self, n, *args):
        """
        Utilises sequence to generate n images.
        """
        pass

    def sequence(self, *args):
        """
        Generate sequence of augmentation functions for image pipeline.
        """
        assert len(args) > 0, "No arguments passed into sequence."

    def rotate(self, img, annots, angle):
        assert angle < 30, "Max angle for image rotation too large."
        out_img, out_annots = self._task(
            img, annots, functions.rotate, angle=angle)
        return out_img, out_annots

    def translate(self, img, annots, du, dv):
        out_img, out_annots = self._task(
            img, annots, functions.translate, du=du, dv=dv)
        return out_img, out_annots

    def flip(self, img, annots, axis):
        out_img, out_annots = self._task(
            img, annots, functions.flip, axis=axis)
        return out_img, out_annots

    def zoom(self, img, annots, fx, fy):
        out_img, out_annots = self._task(
            img, annots, functions.zoom, fx=fx, fy=fy)
        return out_img, out_annots

    def gaussian_blur(self, img, annots, sigma):
        sigma = random.randint(0, int(sigma / 2)) * 2 + 1
        blurred_image = cv2.GaussianBlur(img, (sigma, sigma), 0)
        return blurred_image, annots

    def gaussian_noise(self, img, annots, sigma):
        noise = np.random.normal(0, sigma, (img.shape))
        out_img = img + noise
        out_img = out_img.clip(min=0)
        return out_img, annots

    def _randomise_arguments(self, **kwargs):
        """
        Randomise keyword arguments provided. Will need to add a variable for each potential kwarg
        case possible. This is only required when augmentation failure due to removal of points
        is feasible.
        TODO: Is there a more comprehensive way to do this (eg. cases in swift).
        """
        if not self.randomised:
            return

        randomised_args = {}
        for key, value in kwargs.items():
            # rotation
            if key == "angle":
                new_value = round((random.random() - 0.5) * 2 * value)
            # translation
            if key == "du" or key == "dv":
                new_value = random.randint(-value, value)
            # zoom
            if key == "fx" or key == "fy":
                new_value = (value - 1.) * random.random() + 1.
            randomised_args[key] = new_value

        return randomised_args

    def _task(self, img, annots, func, **kwargs):
        """
        Implement single augmentation function for an array of points.
        """
        original_kwargs = kwargs
        kwargs = self._randomise_arguments(**original_kwargs)

        annots = image_points.homogenous_coordinates(annots)
        out_img, out_annots = func(img, annots, **kwargs)
        within = image_points.are_within_image(out_annots, out_img)

        # repeat if points lost in image
        repeat_attempts = 0
        if not within:
            if not self.randomised:
                raise Exception(
                    'Input augmentation argument always removes points from augmented image.')

            while not within:
                kwargs = self._randomise_arguments(**original_kwargs)
                out_img, out_annots = func(img, annots, **kwargs)
                within = image_points.are_within_image(out_annots, out_img)

                repeat_attempts += 1
                if repeat_attempts > int(1e3):
                    raise Exception(
                        'Too many failed repeat augmentations.')

        return out_img, out_annots
