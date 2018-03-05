#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import xmltodict
import json
from osgeo import gdal
from helpers import lat_long_to_pixel

xml_path = "examples/image.xml"
image_path = "examples/image.jpg"
geojson_path = "examples/buildings.geojson"
segmentation_path = "examples/segmentation.png"


def process_tile(xml_path, image_path, geojson_path, segmentation_path):
    """
    Processes a single tile and returns the corresponding object in
    MS Coco format
    """
    xml = xmltodict.parse(open(xml_path).read())
    tiff_source = gdal.Open(xml["annotation"]["filename"])
    number_of_buildings = len(xml["annotation"]["object"])
    if number_of_buildings == 0:
        "Ignore this"
        # continue

    g = json.loads(open(geojson_path).read())
    _polygons = []
    for f in g["features"]:
        """
        TODO: Assert that this is a Polygon feature
        """
        if (f["type"] == "Feature") and "geometry" in f.keys():
            polygons = f["geometry"]["coordinates"]
            for polygon in polygons:
                _polygon = []
                for coord in polygon:
                    if type(coord) == list and len(coord) == 3:
                            X, Y = lat_long_to_pixel(tiff_source, coord[0], coord[1])
                            _polygon.append([X, Y])
                _polygons.append(_polygon)
                print(_polygon)
        """
        TODO: Aggregate into the correct JSON structure
        """
    print("Locally aggregated Polygons : ", _polygons)
if __name__ == "__main__":
    ms_coco_format = process_tile(xml_path, image_path, geojson_path, segmentation_path)
    print(ms_coco_format)
