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
import uuid
"""
Format floating point values to 2 decimal places
"""
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
# Reference : https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module

np.random.seed(17060728)
random.seed(17060728)

SALT = "5a299fff-89c7-4fda-8d7d-c62b06397919"
OUTPUT = "/mount/SDG/mapping-challenge/generated"
DATASET_NAME = "AOI_2_Vegas_Train"
path = "/mount/SDG/mapping-challenge/{}/PASCALVOC_annotations/annotations/*.jpg".format(DATASET_NAME)

def ensure_directories_exist(dataset_name):
    shutil.rmtree(OUTPUT)
    for mode in ["train", "test"]:
        for dir_type in ["images", "annotations"]:
            try:
                os.makedirs("{}/{}/{}/images".format(OUTPUT, dataset_name, mode))
            except:
                pass

def generate_data(filelist, mode="train"):
    no_buildings = 0
    for _idx, _file in enumerate(filelist):
        print("Failed Processing : {}, {}".format(no_buildings, _idx))
        image_file_name = _file.split("/")[-1]
        image_key = image_file_name.replace(".jpg", "")

        dataset_name = _file.split("/")[-4]
        segcls_path = _file.replace(".jpg", "segcls.png")
        segobj_path = _file.replace(".jpg", "segobj.png")
        xml_path = _file.replace(".jpg", ".xml")
        geojson_path = _file.replace("/annotations/","/geojson/buildings/")\
                            .replace("RGB-PanSharpen", "buildings")\
                            .replace(".jpg", ".geojson")
        # print(image_file_name, "\n", image_key, "\n", xml_path, "\n", \
        #     dataset_name, "\n", xml_path, "\n", geojson_path, "\n", \
        #     segcls_path, "\n", segobj_path)

        image_id = str(uuid.uuid4())
        annotations = process_tile(image_id, xml_path, _file, geojson_path)
        if annotations:
            foo=1
        else:
            no_buildings += 1
            print("No buildings in ", _file)


if __name__ == "__main__":
    # xml_path = "examples/image.xml"
    # image_path = "examples/image.jpg"
    # geojson_path = "examples/buildings.geojson"
    # segmentation_path = "examples/segmentation.png"
    # ms_coco_format = process_tile(54605, xml_path, image_path, geojson_path, rotation=0)
    # print("anns = ",ms_coco_format)

    ensure_directories_exist(DATASET_NAME)
    train_percent = 0.8
    files = glob.glob(path)

    random.shuffle(files)
    marker = int(train_percent*len(files))
    train_set = files[:marker]
    test_set = files[marker:]

    train_annotations = generate_data(train_set, mode="train")
