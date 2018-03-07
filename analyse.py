#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import xmltodict
import json
from osgeo import gdal

SALT = "5a299fff-89c7-4fda-8d7d-c62b06397919"
OUTPUT = "/mount/SDG/mapping-challenge/generated"
path = "/mount/SDG/mapping-challenge/AOI_2_Vegas_Train/PASCALVOC_annotations/annotations/*.xml"
total = 0
empty = 0

np.random.seed(17060728)

import helpers

try:
    shutil.rmtree(OUTPUT)
except:
    pass

for _ann in glob.glob(path):
    # Intialize params
    label_file_name = _ann.split("/")[-1]
    label_key = label_file_name.replace(".xml", "")

    dataset_name = _ann.split("/")[-4]
    segcls_path = _ann.replace(".xml", "segcls.png")
    segobj_path = _ann.replace(".xml", "segobj.png")
    xml_path = _ann.replace(".xml", ".xml")
    geojson_path = _ann.replace("/annotations/","/geojson/buildings/").replace("RGB-PanSharpen", "buildings").replace(".xml", ".geojson")

    print("="*80)
    print(xml_path, "\n", geojson_path, "\n", segcls_path, "\n", segobj_path)
    try:
        os.makedirs(os.path.join(OUTPUT, dataset_name, "train", "images"))
        os.makedirs(os.path.join(OUTPUT, dataset_name, "train", "labels"))
        os.makedirs(os.path.join(OUTPUT, dataset_name, "val", "images"))
        os.makedirs(os.path.join(OUTPUT, dataset_name, "val", "labels"))
        os.makedirs(os.path.join(OUTPUT, dataset_name, "test", "images"))
        os.makedirs(os.path.join(OUTPUT, dataset_name, "test", "labels"))
    except:
        pass
    r = np.random.random()
    if r <= 0.7:
        mode="train"
    elif r <= 0.85:
        mode = "val"
    else:
        mode= "test"
    # Param initializsation done
    xml = xmltodict.parse(open(xml_path).read())
    if "annotation" in xml.keys():
        tiff_source = gdal.Open(xml["annotation"]["filename"])
    else:
        continue

    number_of_buildings = len(xml["annotation"]["object"])
    if number_of_buildings == 0:
        "Ignore this"
        continue

    exit(0)
