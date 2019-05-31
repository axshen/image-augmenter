import random
import cv2
import numpy as np
import glob
import sys
import csv
import argparse

from utils import *
from augmentation_functions import *

class Augmentor:
    def __init__(self, annots_file, image_path, write_path, n_files):
        self.annots_file = annots_file
        self.image_path = image_path
        self.write_path = write_path
        self.n_files = int(n_files)

    def run(self):
        """
        Read annotations from annots_file, with path to images in image_path.
        Writes images (jpg) and annotations (csv) in the write path.
        Specify the number of augmented images to generate with n_files.
        The probability of applying each augmentation and the 'magnitude' of that
        augmentation can be changed in constants below.
        """

        # constants
        max_array = [15, 1.5, 200, 100, 5, 20]
        max_rot, max_zoom, max_shift_u, max_shift_v, max_blur, max_noise = max_array

        probability_array = [1.0, 0.2, 0.4, 1.0, 1.0, 1.0, 0.5]
        p_rot, p_flip_vert, p_flip_horiz, p_zoom, p_shift, p_blur, p_noise = probability_array

        # initialise counter
        count = 0

        # read files
        annotations = []
        with open(self.annots_file,'r') as file:
            reader = csv.reader(file)
            next(reader,None)
            for row in reader:
                annotations.append(row)

        # writing
        new_annots = []
        print('generating images from specified directory')
        while count < self.n_files:
            for row in annotations:
                filename = row[0]
                img = cv2.imread(self.image_path + filename)
                annots = np.array([row[2], row[3], row[4], row[5]]).astype(int)

                aug_img, aug_annots = self.augment(img, annots, max_array, probability_array)

                draw_bounding_box(aug_img, aug_annots, (0,255,0))
                cv2.imshow('Image', aug_img)
                cv2.waitKey(0)

                count += 1

                continue

                # write
                img_name = str(count)+'.jpg'
                cv2.imwrite(self.write_path + img_name, aug_img)
                row = [img_name, 'ball', aug_annots[0], aug_annots[1], aug_annots[2], aug_annots[3]]
                new_annots.append(row)

                # log
                print(row)
                if (count > n_gen):
                    break

                count += 1

        # writing annotations
        with open(self.write_path + "annotations_updated.csv", 'w') as file:
            writer = csv.writer(file, delimiter = ',')
            header = ['filename', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
            writer.writerow(header)
            for row in new_annots:
                writer.writerow(row)

        print('Image augmentation complete.')


    def augment(self, img, annots, max_array, probability_array):

        aug_img = img.copy()
        aug_annots = annots.copy()
        max_rot, max_zoom, max_shift_u, max_shift_v, max_blur, max_noise = max_array
        p_rot, p_flip_vert, p_flip_horiz, p_zoom, p_shift, p_blur, p_noise = probability_array

        # augmentation pipeline
        while np.array_equal(img, aug_img):
            aug_img = img.copy()
            aug_annots = annots.copy()

            # rotation
            prot_gen = random.random()
            if (prot_gen < p_rot):
                aug_img, aug_annots = rotation(img, annots, max_rot)

            # vertical flip
            pflip_gen = random.random()
            if (pflip_gen < p_flip_vert):
                aug_img, aug_annots = flip(aug_img, aug_annots, 0)

            # horizontal flip
            pflip_gen = random.random()
            if (pflip_gen < p_flip_horiz):
                aug_img,aug_annots = flip(aug_img,aug_annots,1)

            # zoom
            pzoom_gen = random.random()
            if (pzoom_gen < p_zoom):
                aug_img, aug_annots = zoom(aug_img, aug_annots, max_zoom)

            # translation
            pshift_gen = random.random()
            if (pshift_gen < p_shift):
                aug_img, aug_annots = translation(aug_img, aug_annots, max_shift_u, max_shift_v)

            # blur
            pblur_gen = random.random()
            if (pblur_gen < p_blur):
                aug_img, aug_annots = blur(aug_img, aug_annots, max_blur)

            # add noise
            pnoise_gen = random.random()
            if (pnoise_gen < p_noise):
                aug_img, aug_annots = random_noise(aug_img, aug_annots, max_noise)

        return aug_img, aug_annots


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help = 'annotations file')
    parser.add_argument('-i', help = 'image path')
    parser.add_argument('-w', help = 'write path (updated images and annotations)')
    parser.add_argument('-n', help = 'number of images to generate')
    args = parser.parse_args()

    annots_file = args.a
    image_path = args.i
    write_path = args.w
    n_files = args.n

    augmentor = Augmentor(annots_file, image_path, write_path, n_files)
    augmentor.run()
