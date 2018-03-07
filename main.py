#!/usr/bin/env python

import glob
from PIL import Image
import numpy as np
import os
import shutil
import xmltodict
import json
from osgeo import gdal
from helpers import lat_long_to_pixel, rotate
from pycocotools import mask as cocomask
import math

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


def process_tile(image_id, xml_path, image_path, geojson_path, segmentation_path, rotation=0):
    """
    Processes a single tile and returns the corresponding object in
    MS Coco format

    :param:
        rotation : Only support rotataion values of 0, 90, -90, 180
    """
    assert rotation in [0, math.pi/2, -1*math.pi/2, math.pi]
    xml = xmltodict.parse(open(xml_path).read())
    tiff_source = gdal.Open(xml["annotation"]["filename"])
    tile_width = int(xml["annotation"]["size"]["width"])
    tile_height = int(xml["annotation"]["size"]["height"])
    number_of_buildings = len(xml["annotation"]["object"])
    if number_of_buildings == 0:
        "Ignore this"
        return False

    annotations = []

    i = 0
    g = json.loads(open(geojson_path).read())

    for f in g["features"]:
        _polygons = []
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
                        if rotation:
                            X, Y = rotate((tile_width/2, tile_height/2), (X, Y), rotation )
                        _polygon.extend([X, Y])

                _polygons.append(_polygon)

            segmentation, bbox, area = get_annotation(_polygons, tile_width, tile_height, poly_format=True)

            annotation = {"segmentation": segmentation,
                          "area": np.float(area),
                          "iscrowd": 0,
                          "image_id": image_id,
                          "bbox": bbox,
                          "category_id": 100,
                          "id": i + 1}

            annotations.append(annotation)

        i = i + 1
    return annotations


def get_annotation(polygons, w, h, poly_format=False):
    segmentation = []
    xmin = w+100
    xmax = -1
    ymin = h+100
    ymax = -1
    for polygon in polygons:
        assert len(polygon) % 2 == 0
        length = int(len(polygon)/2)
        X_ = [polygon[i] for i in range(length)]
        Y_ = [polygon[i*2] for i in range(length)]
        _xmin = min(X_)
        _xmax = max(X_)
        _ymin = min(Y_)
        _ymax = max(Y_)
        if _xmin < xmin: xmin = _xmin
        if _xmax > xmax: xmax = _xmax
        if _ymin < ymin: ymin = _ymin
        if _ymax > ymax: ymax = _ymax
        segmentation.extend(polygon)

    RLEs = cocomask.frPyObjects([segmentation], w, h)
    RLE = cocomask.merge(RLEs)
    area = cocomask.area(RLE)

    bbox = (xmin, ymin, xmax-xmin, ymax-ymin)
    # poly format
    if poly_format:
        return [segmentation], bbox, area
    else:
        # RLE format
        return RLE, bbox, area

if __name__ == "__main__":
    ms_coco_format = process_tile(54605, xml_path, image_path, geojson_path, segmentation_path, rotation=0)
    print("anns = ",ms_coco_format)
