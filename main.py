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
import hashlib
import math

rd = random.Random()
np.random.seed(17060728)
rd.seed(17060728)

SALT = "5a299fff-89c7-4fda-8d7d-c62b06397919"
OUTPUT = "/mount/SDG/mapping-challenge/generated"
DATASET_NAME = "AOI_2_Vegas_Train"
path = "/mount/SDG/mapping-challenge/{}/PASCALVOC_annotations/annotations/*.jpg".format(DATASET_NAME)
IMAGE_PATH_TEMPLATE = "{}/{}/{}images"

image_id_map = {}
def get_random_image_id():
    while True:
        _id = str(uuid.uuid4())[:8]
        # This is just to avoid collision in the random ids
        try:
            foo = image_id_map[_id]
            # If this id exists, then try this again
        except:
            # Else, respond with the generated id
            return _id

def ensure_directories_exist(dataset_name):
    shutil.rmtree(OUTPUT)
    for mode in ["train", "test"]:
        for dir_type in ["images", "annotations"]:
            try:
                os.makedirs("{}/{}/{}/{}".format(OUTPUT, dataset_name, mode, dir_type))
            except:
                pass

def copy_image(source, target_path, rotation):
    assert os.path.exists(source)
    print("Creating File : {} , Rotataion : {}".format(target_path, rotation))
    im = Image.open(source)
    im = im.rotate(rotation)
    im.save(target_path)

def generate_data(filelist, mode="train"):
    no_buildings = 0

    """
    Instantiate dataset json
    """
    dataset = {}
    dataset["info"] = templates.info()
    dataset["categories"] = templates.categories()
    dataset["images"] = []
    dataset["annotations"] = []

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

        image_id = get_random_image_id()

        print(image_id)
        annotations = process_tile(image_id, xml_path, _file, geojson_path)
        if annotations:
            # Copy over file to relevant folder
            target_path = "{}/{}/{}/{}/{}".format(
                OUTPUT, DATASET_NAME, mode,
                "images", image_id+".jpg")
            copy_image(_file, target_path, rotation=0)
            #  Add image entry in dataset json
            dataset["images"].append(
                templates.image(
                    id=image_id,
                    filename=target_path.split("/")[-1],
                    width=400,
                    height=400
                )
            )
            #  Save annotation in dataset json
            dataset["annotations"].append(annotations)

            for rotation in [90, -90, 180]:
                if np.random.random() < 0.2:
                    # Not produce rotated image with a probability of 0.2
                    continue

                image_id = get_random_image_id()
                annotations = process_tile(image_id, xml_path, _file, geojson_path, rotation = 0.5*(rotation*1.0/90)*math.pi)
                # Save Image file
                target_path = "{}/{}/{}/{}/{}".format(
                    OUTPUT, DATASET_NAME, mode,
                    "images", image_id +".jpg")
                copy_image(_file, target_path, rotation=rotation)
                # Add image entry in dataset json
                dataset["images"].append(
                    templates.image(
                        id=image_id,
                        filename=target_path.split("/")[-1],
                        width=400,
                        height=400
                    )
                )

                # Save annotation in dataset json
                dataset["annotations"].append(annotations)
        else:
            no_buildings += 1
            print("No buildings in ", _file)

    # Save dataset annotations
    target_path = "{}/{}/{}/{}/{}".format(
        OUTPUT, DATASET_NAME, mode,
        "annotations", "annotation.json")
    print("Writing dataset annotations to : ", target_path)
    fp = open(target_path, "w")
    fp.write(json.dumps(dataset))
    fp.close()


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
    test_annotations = generate_data(test_set, mode="test")
