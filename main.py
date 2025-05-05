#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import json

import templates
from helpers import process_tile, get_random_id, split_dataset
import random
import json
import uuid
import hashlib
import math
import sys

from loguru import logger
import tqdm
from p_tqdm import p_map

logger.remove() 
logger.add(sys.stdout, level="INFO") 


rd = random.Random()
np.random.seed(17060728)
rd.seed(17060728)

SALT = "5a299fff-89c7-4fda-8d7d-c62b06397919"

DATASET_DIRECTORY = os.getenv("DATASET_DIRECTORY", "/scratch/mohanty/mapping-challenge-data/raw")
OUTPUT_DIRECTORY = os.getenv("OUTPUT_DIRECTORY", "/scratch/mohanty/mapping-challenge-data/processed")
PLACEHOLDER_IMAGES_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, "placeholder_images")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(PLACEHOLDER_IMAGES_DIRECTORY, exist_ok=True)

IMAGE_PATH_TEMPLATE = "{}/{}/{}images"

def ensure_directories_exist(folder_name):
    try:
        shutil.rmtree("{}/{}".format(OUTPUT_DIRECTORY, folder_name))
    except:
        pass
    for dir_type in ["images", "annotations"]:
        try:
            os.makedirs("{}/{}/{}".format(OUTPUT_DIRECTORY, folder_name, dir_type))
        except:
            pass

def copy_image(source, target_path, rotation):
    assert os.path.exists(source)
    logger.debug(f"Creating File : {target_path} , Rotataion : {rotation}")
    im = Image.open(source)
    im = im.rotate(-1 * rotation)
    im.save(target_path)


class MappingChallengeDatasetSplit:
    def __init__(self, dataset_name, split_name, file_list):
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.file_list = file_list
        
        self.image_objects = []
        self.annotation_objects = []
        self.image_id_to_file_path_map = {}
        self.image_id_to_rotation_map = {}
        
        self.num_tiles_without_buildings = 0
        self.num_unique_buildings = 0

        self.load_split()
    
    def parse_image_details_from_filepath(self, filepath):
        image_file_name = filepath.split("/")[-1]
        image_key = image_file_name.replace(".jpg", "")

        dataset_name = filepath.split("/")[-4]
        segcls_path = filepath.replace(".jpg", "segcls.png")
        segobj_path = filepath.replace(".jpg", "segobj.png")
        xml_path = filepath.replace(".jpg", ".xml")
        geojson_path = filepath.replace("/annotations/","/geojson/buildings/")\
                            .replace("RGB-PanSharpen", "buildings")\
                            .replace(".jpg", ".geojson")
        
        return image_file_name, image_key, dataset_name, segcls_path, segobj_path, xml_path, geojson_path
    
    def load_split(self):
        for _idx, _file in enumerate(tqdm.tqdm(self.file_list, desc=f"Loading {self.dataset_name} - {self.split_name}")):
            # Parse Image related details
            image_file_name, image_key, dataset_name, segcls_path, segobj_path, xml_path, geojson_path = \
                self.parse_image_details_from_filepath(_file)

            logger.debug(
                f"Processing {image_file_name}\n{image_key}\n \
                    {xml_path}\n{dataset_name}\n{xml_path}\n \
                    {geojson_path}\n{segcls_path}\n{segobj_path}")

            image_id = get_random_id(mode="image_id") # this function always returns a unique id across all calls
            self.image_id_to_file_path_map[image_id] = _file # keep a map of image id to filepaths
            self.image_id_to_rotation_map[image_id] = 0 # keep a map of image id to rotation
            
            annotations_for_image = process_tile(image_id, xml_path, _file, geojson_path)
            
            if annotations_for_image:            
                # add image and annotations to the internal lists
                self.add_image_and_annotation_objects(image_id, image_file_name, annotations_for_image)
                
                augmentation_source_id = image_id # pass this reference to the augmented images
                # add rotations of the tile to the dataset
                for rotation in [90, -90, 180]:
                    image_id = get_random_id(mode="image_id")
                    self.image_id_to_file_path_map[image_id] = _file
                    self.image_id_to_rotation_map[image_id] = rotation
                
                    annotations_for_image = process_tile(image_id, xml_path, _file, geojson_path, rotation = 0.5*(rotation*1.0/90)*math.pi)
                    self.add_image_and_annotation_objects(image_id, image_file_name, annotations_for_image, augmentation_source_id=augmentation_source_id)
            else:
                self.num_tiles_without_buildings += 1
        
        logger.info(f"Read {len(self.file_list)} tiles with {self.num_unique_buildings} unique buildings and {self.num_tiles_without_buildings} tiles without buildings")
                
    def add_image_and_annotation_objects(self, image_id, image_file_name, annotations_for_image, augmentation_source_id=None):
            self.image_objects.append(
                templates.image(
                    id=image_id,
                    filename=image_file_name,
                    width=300,
                    height=300,
                    augmentation_source_id=augmentation_source_id
                )                    
            )
            self.annotation_objects.extend(annotations_for_image)
            
            self.num_unique_buildings += len(annotations_for_image)
                
    def save_split(self, output_folder_name):
        # Ensure the correct directory structure exists for the said output folder
        ensure_directories_exist(output_folder_name)
        
        logger.info(f"Saving {output_folder_name} split to {OUTPUT_DIRECTORY}/{output_folder_name}")
        
        # for all valid image objects, copy over the image files to the correct directory
        for image_object in tqdm.tqdm(self.image_objects, desc=f"Saving {output_folder_name}"):
            image_id = image_object["id"]
            source_path = self.image_id_to_file_path_map[image_id]
            target_path = f"{OUTPUT_DIRECTORY}/{output_folder_name}/images/{image_id}.jpg"
            image_rotation = self.image_id_to_rotation_map[image_id]
            copy_image(source_path, target_path, rotation=image_rotation)

        # shuffle annotation objects
        random.shuffle(self.annotation_objects)
        
        # write dataset object
        dataset = {
            "info": templates.info(),
            "categories": templates.categories(),
            "images": self.image_objects,
            "annotations": self.annotation_objects
        }
        target_path = f"{OUTPUT_DIRECTORY}/{output_folder_name}/annotations/annotation.json"
        with open(target_path, "w") as fp:
            fp.write(json.dumps(dataset))
        
        logger.info(f"Saved {output_folder_name} split to {target_path}")
        
        # write datamap
        target_path = f"{OUTPUT_DIRECTORY}/{output_folder_name}_image_id_to_file_path_map.json"
        with open(target_path, "w") as fp:
            fp.write(json.dumps(self.image_id_to_file_path_map))
            
    @staticmethod
    def merge_splits(splits, split_name="merged"):
        all_image_objects = []
        all_annotation_objects = []
        all_image_id_to_file_path_map = {}
        all_image_id_to_rotation_map = {}
        total_tiles_without_buildings = 0
        total_unique_buildings = 0

        existing_image_ids = set()
        existing_annotation_ids = set()
        
        for _split in splits:
            all_image_objects.extend(_split.image_objects)
            all_annotation_objects.extend(_split.annotation_objects)
            all_image_id_to_file_path_map.update(_split.image_id_to_file_path_map)
            all_image_id_to_rotation_map.update(_split.image_id_to_rotation_map)
            total_tiles_without_buildings += _split.num_tiles_without_buildings
            total_unique_buildings += _split.num_unique_buildings
            
            # ensure that the image_ids are unique across all splits
            for image_id in _split.image_id_to_file_path_map:
                if image_id in existing_image_ids:
                    raise ValueError(f"Image ID {image_id} is not unique across all splits")
                existing_image_ids.add(image_id)
                
            # ensure that the annotation ids are unique across all splits
            for annotation in _split.annotation_objects:
                if annotation["id"] in existing_annotation_ids:
                    raise ValueError(f"Annotation ID {annotation['id']} is not unique across all splits")
                existing_image_ids.add(annotation["id"])

        # create new split object
        merged_split = MappingChallengeDatasetSplit("merged", split_name, [])
        merged_split.image_objects = all_image_objects
        merged_split.annotation_objects = all_annotation_objects
        merged_split.image_id_to_file_path_map = all_image_id_to_file_path_map
        merged_split.image_id_to_rotation_map = all_image_id_to_rotation_map
        merged_split.num_tiles_without_buildings = total_tiles_without_buildings
        merged_split.num_unique_buildings = total_unique_buildings
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        return merged_split
            
class MappingChallengeDataset:
    def __init__(self, dataset_name, train_percent=0.7, val_percent=0.15, test_percent=0.15):
        
        self.dataset_name = dataset_name
        self.train_percent = train_percent
        self.val_percent = val_percent
        self.test_percent = test_percent
        
        self.create_dataset_splits()
    
    def create_dataset_splits(self):
        image_tile_path_glob = f"{DATASET_DIRECTORY}/{self.dataset_name}/PASCALVOC_annotations/annotations/*.jpg"
        all_files = glob.glob(image_tile_path_glob)
        logger.info(f"Found {len(all_files)} files in dataset {self.dataset_name}")
        
        # shuffle all files
        random.shuffle(all_files)
        
        # create train val and test splits
        self.train_set_files, self.val_set_files, self.test_set_files = \
            split_dataset(all_files, self.train_percent, self.val_percent, self.test_percent)
        
        # create split objects
        self.train_split = MappingChallengeDatasetSplit(self.dataset_name, "train", self.train_set_files)
        self.val_split = MappingChallengeDatasetSplit(self.dataset_name, "val", self.val_set_files)
        self.test_split = MappingChallengeDatasetSplit(self.dataset_name, "test", self.test_set_files)
        
        
        self.train_split.save_split(f"{self.dataset_name}_train")
        self.val_split.save_split(f"{self.dataset_name}_val")
        self.test_split.save_split(f"{self.dataset_name}_test")
        

if __name__ == "__main__":
    datasets = [
                    MappingChallengeDataset("AOI_2_Vegas_Train"),
                    MappingChallengeDataset("AOI_3_Paris_Train"),
                    MappingChallengeDataset("AOI_4_Shanghai_Train"),
                    MappingChallengeDataset("AOI_5_Khartoum_Train")
                ]
    
    merged_train_split = MappingChallengeDatasetSplit.merge_splits(
            [dataset.train_split for dataset in datasets]
    )
    merged_val_split = MappingChallengeDatasetSplit.merge_splits(   
            [dataset.val_split for dataset in datasets]
    )
    merged_test_split = MappingChallengeDatasetSplit.merge_splits(
            [dataset.test_split for dataset in datasets]
    )
    
    merged_train_split.save_split("final/merged_train_split")
    merged_val_split.save_split("final/merged_val_split")
    merged_test_split.save_split("final/merged_test_split")
    
    


# def generate_image_and_annotation_objects(filelist, description=""):
    
#     images = []
#     annotations = []
    
#     for _idx, _file in enumerate(tqdm.tqdm(filelist, desc=description)):
#         # Parse Image related details
#         image_file_name = _file.split("/")[-1]
#         image_key = image_file_name.replace(".jpg", "")

#         dataset_name = _file.split("/")[-4]
#         segcls_path = _file.replace(".jpg", "segcls.png")
#         segobj_path = _file.replace(".jpg", "segobj.png")
#         xml_path = _file.replace(".jpg", ".xml")
#         geojson_path = _file.replace("/annotations/","/geojson/buildings/")\
#                             .replace("RGB-PanSharpen", "buildings")\
#                             .replace(".jpg", ".geojson")

#         logger.debug(
#             f"Processing {image_file_name}\n{image_key}\n \
#                 {xml_path}\n{dataset_name}\n{xml_path}\n \
#                 {geojson_path}\n{segcls_path}\n{segobj_path}")
        
#     #     # Collect a unique image_id
#         image_id = get_random_id(mode="image_id")
#         annotations_for_image = process_tile(image_id, xml_path, _file, geojson_path)
        
        # def add_image_and_annotations_to_dataset(image_id, _file, annotations_for_image, rotation=0):
        #     # Copy over file to a placeholder folder
        #     target_path = f"{PLACEHOLDER_IMAGES_DIRECTORY}/{image_id}.jpg"
        #     copy_image(_file, target_path, rotation=rotation)
        #     images.append(
        #         templates.image(
        #             id=image_id,
        #             filename=target_path.split("/")[-1],
        #             width=300,
        #             height=300,
        #             augmentation_source_id=None
        #         )
        #     )
        #     image_id_map[image_id] = f"{_file}::{rotation}"
        #     annotations.extend(annotations_for_image)
        
#         if annotations_for_image: # If there are annotataions available
#             add_image_and_annotations_to_dataset(image_id, _file, annotations_for_image, rotation=0)            
#             # Add rotations of the tile to the dataset
#             augmentation_source_id = image_id # pass this reference to the augmented images            
#             for rotation in [90, -90, 180]:
#                 image_id = get_random_id(mode="image_id")
#                 annotations = process_tile(image_id, xml_path, _file, geojson_path, rotation = 0.5*(rotation*1.0/90)*math.pi)

#                 # add each copy to the dataset
#                 add_image_and_annotations_to_dataset(image_id, _file, annotations, rotation=rotation)
                
        
        
        
#     # pass

# def add_data_to_data_map(filelist, mode="train"):
#     num_tiles_with_no_buildings = 0
#     total_unique_buildings_detected = 0

#     """
#     Instantiate dataset json
#     """
#     dataset = {}
#     dataset["info"] = templates.info()
#     dataset["categories"] = templates.categories()
#     dataset["images"] = []
#     dataset["annotations"] = []

#     for _idx, _file in enumerate(tqdm.tqdm(filelist, desc=f"Processing {mode} set")):
#         image_file_name = _file.split("/")[-1]
#         image_key = image_file_name.replace(".jpg", "")

#         dataset_name = _file.split("/")[-4]
#         segcls_path = _file.replace(".jpg", "segcls.png")
#         segobj_path = _file.replace(".jpg", "segobj.png")
#         xml_path = _file.replace(".jpg", ".xml")
#         geojson_path = _file.replace("/annotations/","/geojson/buildings/")\
#                             .replace("RGB-PanSharpen", "buildings")\
#                             .replace(".jpg", ".geojson")

#         logger.debug(
#             f"Processing {image_file_name}\n{image_key}\n \
#                 {xml_path}\n{dataset_name}\n{xml_path}\n \
#                 {geojson_path}\n{segcls_path}\n{segobj_path}")

#         image_id = get_random_image_id()

#         annotations = process_tile(image_id, xml_path, _file, geojson_path)
#         if annotations:
#             # Copy over file to relevant folder
#             target_path = f"{OUTPUT_DIRECTORY}/{dataset_name}/{mode}/images/{image_id}.jpg"
            
#             copy_image(_file, target_path, rotation=0)
#             #  Add image entry in dataset json
#             dataset["images"].append(
#                 templates.image(
#                     id=image_id,
#                     filename=target_path.split("/")[-1],
#                     width=400,
#                     height=400,
#                     augmentation_source_id=None
#                 )
#             )
#             # Add to DATA_MAP
#             DATA_MAP[image_id] =  _file + "::0"

#             #  Save annotation in dataset json
#             dataset["annotations"].extend(annotations)
#             total_unique_buildings_detected += len(annotations)
            
#             # Add rotations of the tile to the dataset
#             augmentation_source_id = image_id # pass this reference to the augmented images            
#             for rotation in [90, -90, 180]:
#                 if np.random.random() < 0.2:
#                     # Not produce rotated image with a probability of 0.2
#                     continue

#                 image_id = get_random_image_id()
#                 annotations = process_tile(image_id, xml_path, _file, geojson_path, rotation = 0.5*(rotation*1.0/90)*math.pi)
#                 # Save Image file
#                 target_path = "{}/{}/{}/{}/{}".format(
#                     OUTPUT_DIRECTORY, DATASET_NAME, mode,
#                     "images", image_id +".jpg")
#                 copy_image(_file, target_path, rotation=rotation)
#                 # Add image entry in dataset json
#                 dataset["images"].append(
#                     templates.image(
#                         id=image_id,
#                         filename=target_path.split("/")[-1],
#                         width=300,
#                         height=300,
#                         augmentation_source_id=augmentation_source_id
#                     )
#                 )
#                 # Add to DATA_MAP
#                 DATA_MAP[image_id] =  _file + "::" + str(rotation)

#                 # Save annotation in dataset json
#                 dataset["annotations"].extend(annotations)            
#         else:
#             num_tiles_with_no_buildings += 1
#             logger.debug(f"No buildings in {_file}")

#     logger.info(f"Number of tiles without buildings : {num_tiles_with_no_buildings}")
#     logger.info(f"Total unique buildings detected : {total_unique_buildings_detected}")
#     return dataset
    

# def write_final_dataset(dataset, OUTPIT_FOLDER_NAME="mapping-challenge-v2-aggregated"):
    
#     aggregated_dataset = {}
#     aggregated_dataset["info"] = templates.info()
#     aggregated_dataset["categories"] = templates.categories()
#     aggregated_dataset["images"] = []
#     aggregated_dataset["annotations"] = []
    
    
#     # Correct Image ids
#     print("Correcting Image ids......")
#     for _idx, image in enumerate(dataset["images"]):
#         # Copy file over
#         try:
#             image_id = dataset["images"][_idx]["id"]
#             new_image_id = integral_image_id_map[image_id]
#             source_path = "{}/{}/{}/{}/{}".format(
#                 OUTPUT_DIRECTORY, DATASET_NAME, mode,
#                 "images", image_id+".jpg")
#             target_path = "{}/{}/{}/{}/{}".format(
#                 OUTPUT_DIRECTORY, DATASET_NAME, mode,
#                 "images", str(new_image_id).zfill(12)+".jpg")

#             os.rename(source_path, target_path)
#             dataset["images"][_idx]["id"] = new_image_id
#             dataset["images"][_idx]["file_name"] = str(new_image_id).zfill(12)+".jpg"
#             # Correct entry in DATA_MAP
#             DATA_MAP[new_image_id] = DATA_MAP[image_id]
#             del DATA_MAP[image_id]
#         except Exception as e:
#             print("Encounter Error :( Ignoring")
#             pass


#     # Correct image_ids in Annotations
#     print("Correcting image_ids in annotation files....")
#     for _idx, annotation in enumerate(dataset["annotations"]):
#             image_id = annotation["image_id"]
#             new_image_id = integral_image_id_map[image_id]
#             dataset["annotations"][_idx]["image_id"] = new_image_id

#     # Correct annotation_id in Anotation
#     annotation_ids = []
#     for _idx, annotation in enumerate(dataset["annotations"]):
#             annotation_id = annotation["id"]
#             annotation_ids.append(annotation_id)

#     random.shuffle(annotation_ids)
#     annotation_id_map = {}
#     for _idx, _key in enumerate(annotation_ids):
#         annotation_id_map[_key] = _idx

#     print("Correcting annotation_ids in annotation files....")
#     for _idx, annotation in enumerate(dataset["annotations"]):
#             annotation_id = annotation["id"]
#             new_annotation_id = annotation_id_map[annotation_id]
#             dataset["annotations"][_idx]["id"] = new_annotation_id


#     # Save dataset annotations
#     target_path = "{}/{}/{}/{}/{}".format(
#         OUTPUT, DATASET_NAME, mode,
#         "annotations", "annotation.json")
#     print("Writing dataset annotations to : ", target_path)
#     fp = open(target_path, "w")
#     fp.write(json.dumps(dataset))
#     fp.close()

#     # Save datamap
#     target_path = "{}/{}/{}".format(
#         OUTPUT, DATASET_NAME, mode + "_DATA_MAP.json")
#     fp = open(target_path, "w")
#     fp.write(json.dumps(DATA_MAP))
#     fp.close()
#     print("Writing DATA_MAP to ", target_path)

if __name__ == "__main__1":
    # for _dataset in ["AOI_2_Vegas_Train"]:
    DATA_MAP = {}
    
    TRAIN_VAL_TEST_SPLITS = {}
    
    for _dataset in ["AOI_2_Vegas_Train", "AOI_3_Paris_Train", "AOI_4_Shanghai_Train", "AOI_5_Khartoum_Train"]:
        DATASET_NAME = _dataset
        logger.info("="*80)
        logger.info(f"Processing Dataset: {DATASET_NAME}")
        
        # set_annotation_id_map({})

        # ensure_directories_exist(DATASET_NAME)
        train_percent = 0.7
        val_percent = 0.15
        test_percent = 0.15
        
        
        image_tile_path_glob = f"{DATASET_DIRECTORY}/{DATASET_NAME}/PASCALVOC_annotations/annotations/*.jpg"
        files = glob.glob(image_tile_path_glob)

        random.shuffle(files)
        
        train_set, val_set, test_set = split_dataset(files, train_percent, val_percent, test_percent)
        
        # generate_image_and_annotation_objects(train_set, description="Generating Train Set")
        # generate_image_and_annotation_objects(val_set, description="Generating Validation Set")
        # generate_image_and_annotation_objects(test_set, description="Generating Test Set")
        
        # TRAIN_VAL_TEST_SPLITS[DATASET_NAME] = {
        #     "train": train_set,
        #     "val": val_set,
        #     "test": test_set
        # }

        # train_dataset = add_data_to_data_map(train_set, mode="train")
        # val_dataset = add_data_to_data_map(val_set, mode="val")
        # test_dataset = add_data_to_data_map(test_set, mode="test")