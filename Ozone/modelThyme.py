from PIL import Image
import numpy as np
import pandas as pd
import math
import netCDF4 as nc
from osgeo import gdal

"""
Making a crude model with 3 data sources: traffic, tropomi ozone, and EPA ozone
"""

ds = gdal.Open(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\trafficData\averageTrafficDay.tif", gdal.GA_ReadOnly)
geoTransform = ds.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * ds.RasterXSize
miny = maxy + geoTransform[5] * ds.RasterYSize
extent = [minx, miny, maxx, maxy]


# rb = ds.GetRasterBand(1)
# img_array = rb.ReadAsArray()

fn = r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\gisData\TROPOMI\den.nc\S5P_NRTI_L2__O3_____20230130T204306_20230130T204806_27459_03_020401_20230130T212151\S5P_NRTI_L2__O3_____20230130T204306_20230130T204806_27459_03_020401_20230130T212151.nc"
ds = nc.Dataset(fn)

# extract lat long and ozone data

vars = ds.groups['PRODUCT'].variables
ozone = vars['ozone_total_vertical_column']
lat = vars['latitude']
lon = vars['longitude']

# select area over denver

def ConvertLon(lon):
    lonInEPSG3857 = (lon * 20037508.34 / 180)

    return lonInEPSG3857
def ConvertLat(lat):
    latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)) * (20037508.34 / 180)
    return latInEPSG3857

ConvertLat(39)
ConvertLon(-105)