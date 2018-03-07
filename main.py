#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import json

import templates
from helpers import process_tile


import json
"""
Format floating point values to 2 decimal places
"""
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
# Reference : https://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module

xml_path = "examples/image.xml"
image_path = "examples/image.jpg"
geojson_path = "examples/buildings.geojson"
segmentation_path = "examples/segmentation.png"




if __name__ == "__main__":
    ms_coco_format = process_tile(54605, xml_path, image_path, geojson_path, segmentation_path, rotation=0)
    print("anns = ",ms_coco_format)
