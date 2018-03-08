import glob
from PIL import Image
import numpy as np
import os
import shutil
import xmltodict
import json
from osgeo import gdal
import math

import templates

from pycocotools import mask as cocomask
import uuid
import sys

annotation_id_map = {}
def get_random_annotation_id():
    while True:
        _id = str(uuid.uuid4())[:8]
        # This is just to avoid collision in the random ids
        try:
            foo = annotation_id_map[_id]
            # If this id exists, then try this again
        except:
            # Else, respond with the generated id
            return _id


def process_tile(image_id, xml_path, image_path, geojson_path, rotation=0):
    """
    Processes a single tile and returns the corresponding object in
    MS Coco format

    :param:
        rotation : Only support rotataion values of 0, 90, -90, 180
    """
    assert rotation in [0, math.pi/2, -1*math.pi/2, math.pi]
    xml = xmltodict.parse(open(xml_path).read())
    if "annotation" not in xml.keys():
        return False
    if "object" not in xml["annotation"].keys():
        return False

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

    def gather_polygon(polygon):
        _polygon = []
        for coord in polygon:
            if type(coord) == list and len(coord) == 3:
                X, Y = lat_long_to_pixel(tiff_source, coord[0], coord[1])
                if rotation:
                    X, Y = rotate((tile_width/2, tile_height/2), (X, Y), rotation )
                _polygon.extend([X, Y])
            else:
                raise Exception("Unknown Polyngon recieved : ", polygon)

        return _polygon

    for f in g["features"]:
        _polygons = []
        """
        TODO: Assert that this is a Polygon feature
        """
        if (f["type"] == "Feature") and "geometry" in f.keys() and \
            f["geometry"]["type"] in ["Polygon", "MultiPolygon"]:

            if f["geometry"]["type"] == "Polygon":
                polygons = f["geometry"]["coordinates"]
                for polygon in polygons:
                    _polygons.append(gather_polygon(polygon))
            elif f["geometry"]["type"] == "MultiPolygon":
                multipolygons = f["geometry"]["coordinates"]
                for multipolygon in multipolygons:
                    for polygon in multipolygon:
                        _polygons.append(gather_polygon(polygon))

            segmentation, bbox, area = compute_annotations(
                                            _polygons,
                                            tile_width,
                                            tile_height,
                                            poly_format=False)

            annotation = templates.annotataion(
                                id=get_random_annotation_id(),
                                image_id=image_id,
                                segmentation=segmentation,
                                area=np.float(area),
                                bbox=bbox,
                                category_id=100,
                                is_crowd=0
                                )
            annotations.append(annotation)

        i = i + 1
    return annotations


def compute_annotations(polygons, w, h, poly_format=False):
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
        if sys.version_info >= (3, 0):
            RLE["counts"] = RLE["counts"].decode('ascii')

        return RLE, bbox, area

def lat_long_to_pixel(tiff_source, point_x, point_y, bounds=(0,400, 0, 400)):
    """
    This function assumes xskew and yskew are always 0
    """
    geoTransform = tiff_source.GetGeoTransform()
    width = tiff_source.RasterXSize
    height = tiff_source.RasterYSize
    xOrigin, pixelWidth, xskew, yOrigin, yskew, pixelHeight = geoTransform
    xPixel = int((point_x - xOrigin)*1.0/pixelWidth)
    yPixel = int((point_y - yOrigin)*1.0/pixelHeight)

    minx_ = bounds[0]
    maxx_ = bounds[1]
    miny_ = bounds[2]
    maxy_ = bounds[3]
    xPixel = max(xPixel, minx_)
    xPixel = min(maxx_, xPixel)
    yPixel = max(yPixel, miny_)
    yPixel = min(maxy_, yPixel)
    return xPixel, yPixel

if __name__ == "__main__":
    filepath = "/mount/SDG/mapping-challenge/AOI_2_Vegas_Train/PASCALVOC_annotations/RGB-PanSharpen/RGB-PanSharpen__-115.2454176_36.1563776998.tif"
    tiff_source = gdal.Open(filepath)
    print(lat_long_to_pixel(tiff_source, -115.24491289, 36.156570763000047))

def rotate(origin, point, angle, integral=False):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    assert angle in [0, math.pi/2, -1*math.pi/2, math.pi]
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

    if integral:
        qx = int(qx)
        qy = int(qy)
    return qx, qy
