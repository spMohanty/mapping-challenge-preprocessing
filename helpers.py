from osgeo import gdal
import numpy as np

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
