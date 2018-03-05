from osgeo import gdal, osr
import numpy as np
import json
import sys
from shapely.geometry import shape, Point
from PIL import Image

import numpy as np
from osgeo import gdal

path_to_tiff_foder = '/mount/SDG/sean/spacenet/data/processedBuildingLabels/3band/'
path_to_geojson_folder = '/mount/SDG/sean/spacenet/data/processedBuildingLabels/vectorData/geoJson/'
def get_coordinates(filepath):
	_file = filepath
	ds = gdal.Open(_file)
	geoTransform = ds.GetGeoTransform()
	width = ds.RasterXSize
	height = ds.RasterYSize
	xOrigin = geoTransform[0]
	yOrigin = geoTransform[3]
	pixelWidth = geoTransform[1]
	pixelHeight = geoTransform[5]
	coords = np.zeros((width,height,2))
	for i in range(0,width):
		for j in range(0,height):
			coords[i][j][0] = (i * pixelWidth) + xOrigin
			coords[i][j][1] = (j * pixelHeight) + yOrigin
	return coords

filename = path_to_tiff_foder + '3band_AOI_1_RIO_img2083.tif'
fake_coord = get_coordinates(filename) # 438x406 list of lists // np array of size (43,406,2)

img = np.zeros((439, 406, 3), dtype=np.uint8)
points = []
polygons = []

print(fake_coord.shape)
for _x in range(fake_coord.shape[0]):
	for _y in range(fake_coord.shape[1]):
#		print(_x, _y)
		coord = fake_coord
		points.append(Point(coord[_x, _y, 0], coord[_x, _y, 1]))


geojson_file = path_to_geojson_folder+'/Geo_AOI_1_RIO_img2083.geojson'
with open(geojson_file) as f:
        data = json.load(f)

#for feature in data['features']:
#   print feature['geometry']['type']
#    print feature['geometry']['coordinates']

for feature in data['features']:
    if feature['geometry']['type'] == 'Polygon':
        polygon = shape(feature['geometry'])
        polygons.append(polygon)

ds = gdal.Open(filename)
gt = ds.GetGeoTransform()
xorig = gt[0]
yorig = gt[3]
pw = gt[1]
ph = gt[5]

for index, point in enumerate(points):
    for polygon in polygons:
        if polygon.contains(point):
		px = point.x
		py = point.y
		xco = int((px - xorig)/pw)
		yco = int((py - yorig)/ph)
        	print("Found Point : ", xco, yco)
        	print("Found Point__RAW : ", px, py)
#		print(img.shape)
		img[xco][yco][:] = 0
		img[xco][yco][0] = 255
img = np.ascontiguousarray(img.transpose(2, 0, 1))

import visdom
vis = visdom.Visdom()
vis.image(img, win="mapping", env="crowdai")

picture = Image.open(filename)

picture = np.asarray(picture)
print(picture.shape)
picture = np.ascontiguousarray(picture.transpose(2, 1, 0))
vis.image(picture, win="picture", env="crowdai")

#random_image = np.random.rand(3, 256, 256)
#vis.image(random_image, win="random", env="crowdai")

#img = Image.fromarray(img, 'RGB')
#print(img)
#img.save('my.png')
#img.show()
