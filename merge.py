#!/usr/bin/env python
import json
import shutil
import random
import glob
import os
from pycocotools.coco import COCO
import templates
import random

for mode in ["train", "val", "test"]:
    OUTPUT_PATH = "../../generated/FINAL/{}".format(mode)
    OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_PATH, "images")
    try:
        shutil.rmtree(OUTPUT_PATH)
    except:
        pass
    os.mkdir(OUTPUT_PATH)
    os.mkdir(OUTPUT_IMAGES_DIR)


    dataset_names = ["AOI_2_Vegas_Train", "AOI_3_Paris_Train", "AOI_4_Shanghai_Train", "AOI_5_Khartoum_Train"]
    path_template = "../../generated/{}/"
    dataset_paths = [path_template.format(x) for x in dataset_names]

    _dict = {}

    WIDTH=300
    HEIGHT=300

    total_images = 0
    ALL_IMAGES = []
    ALL_PATHS = []

    for path in dataset_paths:
        images_dir = os.path.join(path, mode, "images")
        annotations_path = os.path.join(path, mode, "annotations", "annotation.json")
        print(annotations_path)
        annotations = COCO(annotations_path)
        _dict[path] = annotations
        image_ids = list(annotations.imgs.keys())
        ALL_IMAGES = ALL_IMAGES + list(image_ids)
        ALL_PATHS = ALL_PATHS + ([path]*len(image_ids))

    # Dict prepared
    print("Shuffling All IDS")
    INDICES = list(range(len(ALL_IMAGES)))
    random.shuffle(INDICES)

    FINAL_IMAGES = []
    FINAL_ANNOTATIONS = []

    SMALL_ANNOTATIONS = []
    SMALL_IMAGES = []

    annotation_id = 0
    new_id = -1
    for _idx in INDICES:
        path = ALL_PATHS[_idx]
        images_dir = os.path.join(path, mode, "images")
        coco = _dict[path]

        old_id = ALL_IMAGES[_idx]
        new_id += 1
        print(path, old_id, new_id)

        # Copy file to output directory
        old_filename = str(old_id).zfill(12)+".jpg"
        old_path = os.path.join(images_dir, old_filename)
        new_filename = str(new_id).zfill(12)+".jpg"
        new_path = os.path.join(OUTPUT_IMAGES_DIR, new_filename)
        try:
            shutil.copy(
                old_path,
                new_path
            )
        except:
            continue
        print(new_id, new_filename)
        image_template = templates.image(new_id, new_filename, width=300, height=300)
        FINAL_IMAGES.append(image_template)

        # Correct and add annotations
        annotation_ids = coco.getAnnIds(imgIds=old_id)
        IMAGE_ANNOTATIONS = []
        for _ann_id in annotation_ids:
            ann = coco.loadAnns(_ann_id)
            for _ann in ann:
                _ann["id"] = annotation_id
                annotation_id += 1
                _ann["image_id"] = new_id
                IMAGE_ANNOTATIONS.append(_ann)
        FINAL_ANNOTATIONS.extend(IMAGE_ANNOTATIONS)

        if random.randint(0, 100) <= 2:
            SMALL_ANNOTATIONS.extend(IMAGE_ANNOTATIONS)
            SMALL_IMAGES.append(image_template)

    print("Doing the final shuffle.....")
    random.shuffle(FINAL_IMAGES)
    random.shuffle(FINAL_ANNOTATIONS)
    random.shuffle(SMALL_IMAGES)
    random.shuffle(SMALL_ANNOTATIONS)

    final_object = {}
    final_object["info"] = templates.info()
    final_object["categories"] = templates.categories()
    final_object["images"] = FINAL_IMAGES
    final_object["annotations"] = FINAL_ANNOTATIONS

    print("Writing to json")
    annotation_output_path = os.path.join(OUTPUT_PATH,"annotation.json")
    fp = open(annotation_output_path, "w")
    fp.write(json.dumps(final_object))
    fp.close()

    # Write small annotations
    small_object = {}
    small_object["info"] = templates.info()
    small_object["categories"] = templates.categories()
    small_object["images"] = SMALL_IMAGES
    small_object["annotations"] = SMALL_ANNOTATIONS
    print("Writing to json")
    annotation_output_path = os.path.join(OUTPUT_PATH,"annotation-small.json")
    fp = open(annotation_output_path, "w")
    fp.write(json.dumps(small_object))
    fp.close()
