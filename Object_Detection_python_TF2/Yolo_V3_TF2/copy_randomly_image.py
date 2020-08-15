#!/usr/bin/env python3

"""
Author: David Valencia
Date: 12 / May / 2020

Description:  Sometimes the dataset is huge (for example 25K images) and we only need part
			  of those images (for example 1k image), this script randomly takes the wanted
			  number of images and copies them to another directory.
			  Also, I only select the images which size is bigger than 600 x 600 in order to
			  avoid small and irrelevante images

Note:         Could happen that there will be less image that the number_select_images
			  this is because the random function sometimes  repeats a number
"""

import os
import cv2
import shutil
import random

path_input = '/home/david/MAFA_mask_Dataset/test-images/iages/'
path_ouput = '/home/david/PycharmProjects/datasets/Human_Face_Mask/test/iages/'

number_select_images = 130
process = True
image_counter = 0

while process == True:

    index = random.choice(os.listdir(path_input))
    file_to_copy = (path_input + index)
    img = cv2.imread(file_to_copy)
    w = img.shape[0]
    h = img.shape[1]

    if w and h >= 600:
        image_counter += 1
        shutil.copy(file_to_copy, path_ouput)
        print(file_to_copy)
    if image_counter == number_select_images:
        break

"""   
for i in range(1, number_select_images + 1):
    index = random.choice(os.listdir(path_input))
    file_to_copy = (path_input + index)
    img = cv2.imread(file_to_copy)
    w = img.shape[0]
    h = img.shape[1]
    if w and h >= 400:
        shutil.copy(file_to_copy, path_ouput)
        print(file_to_copy)
"""
print(f"Done, {number_select_images} images have been copied to the select output path")
