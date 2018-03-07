from osgeo import gdal
import numpy as np
import math

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
    # xPixel = max(xPixel, minx_)
    # xPixel = min(maxx_, xPixel)
    # yPixel = max(yPixel, miny_)
    # yPixel = min(miny_, yPixel)
    return xPixel, yPixel

if __name__ == "__main__":
    filepath = "/mount/SDG/mapping-challenge/AOI_2_Vegas_Train/PASCALVOC_annotations/RGB-PanSharpen/RGB-PanSharpen__-115.2454176_36.1563776998.tif"
    tiff_source = gdal.Open(filepath)
    print(lat_long_to_pixel(tiff_source, -115.24491289, 36.156570763000047))

def rotate(origin, point, angle, integral=True):
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
