"""
Author: David Valencia
Date: 28 / May / 2020
Description:  Convert file.txt with YOLO label format to file.xml with Pascal VOC  label format
			 The input folder has the training images and the yolo label of each image
			 In the output folder a xml file will be created for each image of the input folder
"""

import os
import cv2
import xml.etree.cElementTree as ET

# -------------- Input/Output Paths-------------------#

# For human face data set
# path_images = '/home/david/PycharmProjects/datasets/Human_Face/train/images/' # path the image data_set
# path_labels = "/home/david/PycharmProjects/datasets/Human_Face/train/labels_YOLO/"  # path with the YOLO labels
# path_output = '/home/david/PycharmProjects/datasets/Human_Face/train/labels_pascal_voc/' # path of the new labels with pascal VOC format


# For human face with mask  data set
path_images = '/home/david/PycharmProjects/datasets/Human_Face_Mask/train/images/'  # path the image data_set
path_labels = "/home/david/PycharmProjects/datasets/Human_Face_Mask/train/labels_YOLO/"  # path with the YOLO labels
path_output = '/home/david/PycharmProjects/datasets/Human_Face_Mask/train/labels_pascal_voc/'  # path of the new labels with pascal VOC format

class_name = {'0': 'Face_no_mask', '1': 'Face_with_mask'}


def create_root(file_prefix, width, height):
    root = ET.Element("annotations")

    ET.SubElement(root, "filename").text = "{}.jpg".format(file_prefix)
    ET.SubElement(root, "folder").text = "Database_Face_Mask"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    return root


def create_object_annotation(root, voc_labels):
    for voc_label in voc_labels:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = voc_label[0]
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = str(0)
        ET.SubElement(obj, "difficult").text = str(0)

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(voc_label[1])
        ET.SubElement(bbox, "ymin").text = str(voc_label[2])
        ET.SubElement(bbox, "xmax").text = str(voc_label[3])
        ET.SubElement(bbox, "ymax").text = str(voc_label[4])

    return root


def write_xml_file(file_prefix, w, h, voc_labels):
    root = create_root(file_prefix, w, h)
    root = create_object_annotation(root, voc_labels)

    tree = ET.ElementTree(root)
    tree.write("{}{}.xml".format(path_output, file_prefix))


def read_files():
    if not os.path.exists(path_output):
        print("there is no output folder")

    for filename in os.listdir(path_labels):

        if filename.endswith('txt'):

            file_prefix = filename.split(".txt")[0]

            if file_prefix == "classes":
                pass

            else:

                image_file_name = "{}.jpg".format(file_prefix)
                img = cv2.imread(path_images + image_file_name)

                w = img.shape[1]
                h = img.shape[0]

                with open(path_labels + filename, 'r') as file:

                    lines = file.readlines()
                    voc_labels = []

                    for line in lines:
                        voc = []
                        line = line.strip()
                        data = line.split()

                        voc.append(class_name.get(data[0]))

                        bbox_width = float(data[3]) * w
                        bbox_height = float(data[4]) * h
                        center_x = float(data[1]) * w
                        center_y = float(data[2]) * h

                        voc.append(center_x - (bbox_width / 2))
                        voc.append(center_y - (bbox_height / 2))
                        voc.append(center_x + (bbox_width / 2))
                        voc.append(center_y + (bbox_height / 2))

                        voc_labels.append(voc)

                    write_xml_file(file_prefix, w, h, voc_labels)


read_files()