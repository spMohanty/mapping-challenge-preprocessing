#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import json

import templates
from helpers import process_tile, get_annotation_id_map, set_annotation_id_map, split_dataset
import random
import json
import uuid
import hashlib
import math

from loguru import logger
import tqdm
from p_tqdm import p_map

rd = random.Random()
np.random.seed(17060728)
rd.seed(17060728)

SALT = "5a299fff-89c7-4fda-8d7d-c62b06397919"

DATASET_DIRECTORY = os.getenv("DATASET_DIRECTORY", "/scratch/mohanty/mapping-challenge-data/raw")
OUTPUT = os.getenv("OUTPUT_DIRECTORY", "/scratch/mohanty/mapping-challenge-data/processed")
os.makedirs(OUTPUT, exist_ok=True)

DATASET_NAME = os.getenv("DATASET_NAME", "AOI_2_Vegas_Train")

# path = "/mount/SDG/mapping-challenge/{}/PASCALVOC_annotations/annotations/*.jpg".format(DATASET_NAME)

IMAGE_PATH_TEMPLATE = "{}/{}/{}images"

DATA_MAP = {}

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
    try:
        shutil.rmtree("{}/{}".format(OUTPUT, dataset_name))
    except:
        pass
    for mode in ["train", "val", "test"]:
        for dir_type in ["images", "annotations"]:
            try:
                os.makedirs("{}/{}/{}/{}".format(OUTPUT, dataset_name, mode, dir_type))
            except:
                pass

def copy_image(source, target_path, rotation):
    assert os.path.exists(source)
    logger.debug(f"Creating File : {target_path} , Rotataion : {rotation}")
    im = Image.open(source)
    im = im.rotate(-1 * rotation)
    im.save(target_path)

def generate_data(filelist, mode="train"):
    num_tiles_with_no_buildings = 0
    total_unique_buildings_detected = 0

    """
    Instantiate dataset json
    """
    dataset = {}
    dataset["info"] = templates.info()
    dataset["categories"] = templates.categories()
    dataset["images"] = []
    dataset["annotations"] = []

    for _idx, _file in enumerate(tqdm.tqdm(filelist, desc=f"Processing {mode} set")):
        image_file_name = _file.split("/")[-1]
        image_key = image_file_name.replace(".jpg", "")

        dataset_name = _file.split("/")[-4]
        segcls_path = _file.replace(".jpg", "segcls.png")
        segobj_path = _file.replace(".jpg", "segobj.png")
        xml_path = _file.replace(".jpg", ".xml")
        geojson_path = _file.replace("/annotations/","/geojson/buildings/")\
                            .replace("RGB-PanSharpen", "buildings")\
                            .replace(".jpg", ".geojson")

        logger.debug(
            f"Processing {image_file_name}\n{image_key}\n \
                {xml_path}\n{dataset_name}\n{xml_path}\n \
                {geojson_path}\n{segcls_path}\n{segobj_path}")

        image_id = get_random_image_id()

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
                    height=400,
                    augmentation_source_id=None
                )
            )
            # Add to DATA_MAP
            DATA_MAP[image_id] =  _file + "::0"

            #  Save annotation in dataset json
            dataset["annotations"].extend(annotations)
            total_unique_buildings_detected += len(annotations)
            
            # Add rotations of the tile to the dataset
            augmentation_source_id = image_id # pass this reference to the augmented images            
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
                        width=300,
                        height=300,
                        augmentation_source_id=augmentation_source_id
                    )
                )
                # Add to DATA_MAP
                DATA_MAP[image_id] =  _file + "::" + str(rotation)

                # Save annotation in dataset json
                dataset["annotations"].extend(annotations)            
        else:
            num_tiles_with_no_buildings += 1
            logger.debug(f"No buildings in {_file}")

    logger.info(f"Number of tiles without buildings : {num_tiles_with_no_buildings}")
    logger.info(f"Total unique buildings detected : {total_unique_buildings_detected}")
    # Convert image_id and annotation id to integers (as coco api expects them to be)
    image_ids = list(DATA_MAP.keys())
    random.shuffle(image_ids)
    integral_image_id_map = {}
    for _idx, _image_id in enumerate(image_ids):
        integral_image_id_map[_image_id] = _idx

    # Correct Image ids
    print("Correcting Image ids......")
    for _idx, image in enumerate(dataset["images"]):
        # Copy file over
        try:
            image_id = dataset["images"][_idx]["id"]
            new_image_id = integral_image_id_map[image_id]
            source_path = "{}/{}/{}/{}/{}".format(
                OUTPUT, DATASET_NAME, mode,
                "images", image_id+".jpg")
            target_path = "{}/{}/{}/{}/{}".format(
                OUTPUT, DATASET_NAME, mode,
                "images", str(new_image_id).zfill(12)+".jpg")

            os.rename(source_path, target_path)
            dataset["images"][_idx]["id"] = new_image_id
            dataset["images"][_idx]["file_name"] = str(new_image_id).zfill(12)+".jpg"
            # Correct entry in DATA_MAP
            DATA_MAP[new_image_id] = DATA_MAP[image_id]
            del DATA_MAP[image_id]
        except Exception as e:
            print("Encounter Error :( Ignoring")
            pass


    # Correct image_ids in Annotations
    print("Correcting image_ids in annotation files....")
    for _idx, annotation in enumerate(dataset["annotations"]):
            image_id = annotation["image_id"]
            new_image_id = integral_image_id_map[image_id]
            dataset["annotations"][_idx]["image_id"] = new_image_id

    # Correct annotation_id in Anotation
    annotation_ids = []
    for _idx, annotation in enumerate(dataset["annotations"]):
            annotation_id = annotation["id"]
            annotation_ids.append(annotation_id)

    random.shuffle(annotation_ids)
    annotation_id_map = {}
    for _idx, _key in enumerate(annotation_ids):
        annotation_id_map[_key] = _idx

    print("Correcting annotation_ids in annotation files....")
    for _idx, annotation in enumerate(dataset["annotations"]):
            annotation_id = annotation["id"]
            new_annotation_id = annotation_id_map[annotation_id]
            dataset["annotations"][_idx]["id"] = new_annotation_id


    # Save dataset annotations
    target_path = "{}/{}/{}/{}/{}".format(
        OUTPUT, DATASET_NAME, mode,
        "annotations", "annotation.json")
    print("Writing dataset annotations to : ", target_path)
    fp = open(target_path, "w")
    fp.write(json.dumps(dataset))
    fp.close()

    # Save datamap
    target_path = "{}/{}/{}".format(
        OUTPUT, DATASET_NAME, mode + "_DATA_MAP.json")
    fp = open(target_path, "w")
    fp.write(json.dumps(DATA_MAP))
    fp.close()
    print("Writing DATA_MAP to ", target_path)

if __name__ == "__main__":
    # for _dataset in ["AOI_2_Vegas_Train"]:
    for _dataset in ["AOI_2_Vegas_Train", "AOI_3_Paris_Train", "AOI_4_Shanghai_Train", "AOI_5_Khartoum_Train"]:
        DATASET_NAME = _dataset
        logger.info("="*80)
        logger.info(f"Processing Dataset: {DATASET_NAME}")
        
        DATA_MAP = {}
        set_annotation_id_map({})

        ensure_directories_exist(DATASET_NAME)
        train_percent = 0.7
        val_percent = 0.15
        test_percent = 0.15
        
        
        image_tile_path_glob = f"{DATASET_DIRECTORY}/{DATASET_NAME}/PASCALVOC_annotations/annotations/*.jpg"
        files = glob.glob(image_tile_path_glob)

        random.shuffle(files)
        
        train_set, val_set, test_set = split_dataset(files, train_percent, val_percent, test_percent)

        train_annotations = generate_data(train_set, mode="train")
        train_annotations = generate_data(val_set, mode="val")
        test_annotations = generate_data(test_set, mode="test")
