from PIL import Image
import numpy as np
import pandas as pd
import math
import netCDF4 as nc
from osgeo import gdal

"""
Making a crude model with 3 data sources: traffic, tropomi ozone, and EPA ozone
"""

# Making a traffic raster
im = Image.open(r"/DU-Thesis/Ozone/trafficRasters/averageTrafficDay1.tif")
traffic = np.array(im)

oneHourAvg = traffic * 1/24
tfWd = {}
times = list(range(0,24))
points = 0

for time in times:
    if time >= 0 and time < 6:
        tfWd['traffic{}'.format(time)] = oneHourAvg * 0.10
        points += 0.10
        print(points)
    elif time >= 6 and time < 10:
        tfWd['traffic{}'.format(time)] = oneHourAvg * 2
        points += 2
        print(points)
    elif time >= 10 and time <15:
        tfWd['traffic{}'.format(time)] = oneHourAvg * 1
        points += 1
        print(points)
    elif time >= 15 and time < 19:
        tfWd['traffic{}'.format(time)] = oneHourAvg * 2
        points += 2
        print(points)
    elif time == 23:
        tfWd['traffic{}'.format(time)] = oneHourAvg * 0.4
        points += 0.4
        print(points)
    else:
        tfWd['traffic{}'.format(time)] = oneHourAvg * 0.5
        points += 0.5
        print(points)

ds = gdal.Open(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\trafficRasters\averageTrafficDay1.tif", gdal.GA_ReadOnly)
geoTransform = ds.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * ds.RasterXSize
miny = maxy + geoTransform[5] * ds.RasterYSize
extent = [miny, minx, maxy, maxx]


denverOzone['LULC'] = []
type(denverOzoneJuly)