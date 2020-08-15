#!/usr/bin/env python3

"""
Author: David Valencia
Date: 10 / May / 2020
Description:  Open Image V4 format annotation labels to YOLO format annotation labels


Firstly, it is necessary to download a considerable number of images related to the object that we want to detect,
for example, bus, car, human face. An easily accessible repository is Open Images Dataset V4 from Google.
Available in https://storage.googleapis.com/openimages/web/index.html. Open Images is a dataset of ~9M images that
have been annotated with image-level labels and object bounding boxes. The training set of V4 contains 14.6M bounding
boxes for 600 object classes on 1.74M images

However, this dataset contains more than 600 classes with many Gb of pictures but we don't want to use nor download
the entire dataset, only the images and labels that we are needing.

To download images that contain only specific classes of the dataset objects we can use the OIDv4 Toolkit
available at https://github.com/pythonlessons/OIDv4_ToolKit . To download image with OIDv4:

    Clone the repo to whatever directory you want:
        * git clone https://github.com/EscVM/OIDv4_ToolKit.git
        * we can run pip3 install -r requirements.txt , however I prefer check the "requirements.txt" file for the
          dependencies and install each manually, skipping things you know you already have.

    To download the images:

        I recommend, first look for the correct name of the class we want to download, go to the
        Open Image Dataset page, click explore, then in the Type box select: Detection, then in
        the category box find the class we want for example: Human face and let's see the results.

        With the help of OID4 we are going to download the Human face category. Go to into the
        directory where we cloned OIDv4 and type:

         * python3 main.py downloader --classes Human_face --type_csv train --limit 500
         * python3 main.py downloader --classes Human_face --type_csv test --limit 100
         * python3 main.py downloader --classes Human_face --type_csv validation --limit 100

        Also you can try:
         -- type_csv all  to download all 3 folders at once
         -- limit 500 limit the number of images to download

        If it is the first time we are running OID4 we will probably have the following errors:

         [ERROR] Missing the train-annotations-bbox.csv file.
         [DOWNLOAD] Do you want to download the missing file? [Y/n]

         Answer Y to all 3 of these, whenever you encounter them (it should prompt just before it
         downloads the training, test, and validation CSV files). These files it asks to download are the CSV files
         (that I will refer to as 'annotation CSVs') that contain the information about where
         the objects are (the bounding boxes of the objects) in the corresponding photos we're downloading.

         After this, you should see some bars with which indicates the downloads process
         If you downloaded Human_face class, in your cloned the OIDv4Toolkit  folder, you should
         have a directory structure similar to:

        OIDv4_ToolKit
        │   main.py
        │   LICENSE
        │   README.md
        └───OID
            │
            └───csv_folder
            │    │   class-descriptions-boxable.csv
            │    │   validation-annotations-bbox.csv
            │	 │	 test-annotations-bbox.csv
            │	 │	 train-annotations-bbox.csv
            └───Dataset
                │
                └─── test
                     │
                     └─── Human face
                           │
                           │xxxxxxxx.jpg
                           │xxxxxxxy.jpg
                           │xxxxxxxz.jpg
                           │...
                           └───Labels
                                  │
                                  │xxxxxxxx.txt
                                  │xxxxxxxy.txt
                                  │xxxxxxxz.txt
                                  │...
                │
                └─── train
                     │
                     └───Human face
                           │
                           │.... (same idea as above)
                │
                └─── validation
                     │ .... (same idea as above)


Now we are going to create the proper txt files for each image i.e the labels in YOLO format

            Remember,this script only preps training data for one class of object at a time
            Here is what sets me apart from the rest, to convert labels to the correct YOLO
            format many use thefile train-annotations-bbox.csv that contains the information
            of each image, but I think there is a faster and easier way.

            I use the Labels folder that is downloaded together with the images, i.e.,
            I use the folder:OIDv4_ToolKit/OID/Dataset/train/Labels.

            Originally the annotation for each bounding box has the structure:
                 Human face 440.32 212.480256 446.72 227.199744
                 Human face 451.84 134.39999999999998 481.28 178.56

            After going through the python script the YOLO format for each bounding box will be:
                0 0.6059375 0.1975 0.026874999999999982 0.05166599999999999
                0 0.8103125 0.445833 0.024375000000000036 0.03499999999999999

            This script changes all your labels to the correct format needed in YOLO,
            just remember to assign the correct paths.
"""
import cv2
from os import walk

# -------------- Input/Output Paths -------------------#
path_images = '/home/david/PycharmProjects/datasets/Human_Face/train/images/' # path the image data_set
path_labels = "/home/david/PycharmProjects/datasets/Human_Face/train/labels_Original/"  # path with the original labels
path_output = '/home/david/PycharmProjects/datasets/Human_Face/train/labels_YOLO/' # path of the new labels with YOLO format

# -------Create the necessary file --> classes.txt-----#
class_name = "Human_Face"
class_number = 0
classes_file_outpath = path_output + "classes.txt"
classes_file = open(classes_file_outpath, "w")
classes_file.write(class_name)

# ------ Get the name of the label files and images---#
labels_name_list = []
for (dirpath, dirnames, filenames) in walk(path_labels):
    labels_name_list.extend(sorted(filenames))
    break

images_name_list = []
for (dirpath, dirnames, filenames) in walk(path_images):
    images_name_list.extend(sorted(filenames))
    break

# ---------- Convert the labels to the proper format -----#
def get_text_original_label():

    for i in range(len(labels_name_list)):

        img = cv2.imread(path_images + images_name_list[i])
        w = img.shape[0]
        h = img.shape[1]

        txt_file = open(path_labels + labels_name_list[i], "r")
        lines = txt_file.read().splitlines()

        txt_outpath = path_output + labels_name_list[i]
        txt_outfile = open(txt_outpath, "w")

        for line in lines:
            elements = line.split(' ')
            x_min = float(elements[len(elements) - 4])
            y_min = float(elements[len(elements) - 3])
            x_max = float(elements[len(elements) - 2])
            y_max = float(elements[len(elements) - 1])

            yolo_format = convert_yolo_label(w, h, x_min, y_min, x_max, y_max)
            txt_outfile.write(str(class_number) + " " + " ".join([str(a) for a in yolo_format]) + '\n')

    txt_file.close()
    txt_outfile.close()
    classes_file.close()


def convert_yolo_label(size_w, size_h, x_min, y_min, x_max, y_max):
    dw = 1. / size_h
    dh = 1. / size_w
    x = (x_min + x_max) / 2.0
    y = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

get_text_original_label()
