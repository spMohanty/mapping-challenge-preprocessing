#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import json

import templates
from helpers import process_tile
import random
import json
"""
Format floating point values to 2 decimal places
"""
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
# Reference : https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module

np.random.seed(17060728)
random.seed(17060728)

xml_path = "examples/image.xml"
image_path = "examples/image.jpg"
geojson_path = "examples/buildings.geojson"
segmentation_path = "examples/segmentation.png"

def generate_data(filelist):
    for _file in filelist:
        print(_file)
        break

if __name__ == "__main__":
    # ms_coco_format = process_tile(54605, xml_path, image_path, geojson_path, segmentation_path, rotation=0)
    # print("anns = ",ms_coco_format)

    SALT = "5a299fff-89c7-4fda-8d7d-c62b06397919"
    OUTPUT = "/mount/SDG/mapping-challenge/generated"
    path = "/mount/SDG/mapping-challenge/AOI_2_Vegas_Train/PASCALVOC_annotations/annotations/*.xml"
    train_percent = 0.8
    files = glob.glob(path)

    random.shuffle(files)
    marker = int(train_percent*len(files))
    train_set = files[:marker]
    test_set = files[marker:]

    train_annotations = generate_data(train_set)
