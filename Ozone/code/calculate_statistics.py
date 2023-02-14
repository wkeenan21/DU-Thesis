import pandas as pd
import numpy as np
#from PIL import Image
from osgeo import gdal
# import data
OzoneJuly = pd.read_csv (r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyOzone.csv")
ds = gdal.Open(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\gisData\LULC\rc_rc_clip.tif", gdal.GA_ReadOnly)
LULC = np.array(ds.GetRasterBand(1).ReadAsArray())
geoTransform = ds.GetGeoTransform()
minx = geoTransform[0]
maxy = geoTransform[3]
maxx = minx + geoTransform[1] * ds.RasterXSize
miny = maxy + geoTransform[5] * ds.RasterYSize
extent = [miny, minx, maxy, maxx]

latitudes = OzoneJuly['latitude'].unique()
longitudes = OzoneJuly['longitude'].unique()
sites = OzoneJuly['site_number'].unique()

coords = []
for i in range(len(latitudes)):
    coords.append((latitudes[i], longitudes[i]))

O3J = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyPlus.csv")

# make new column for integer time
O3J['local_time_int'] = O3J['time_local']
O3J['local_time_int'] = O3J['local_time_int'].str.replace(':00', '')
O3J['local_time_int'] = pd.to_numeric(O3J['local_time_int'])

O3J.loc[O3J['local_time_int'] < 6, O3J['averageTra']] = O3J['averageTra'] * 0.1
# make a new column
O3J['traffic_hour'] = np.zeros(8184)
# assign
O3J['traffic_hour'] = O3J.loc[O3J['local_time_int'] < 6]['averageTra']*0.1
O3J['traffic_hour'] = O3J.loc[(O3J['local_time_int'] >= 6 and O3J['local_time_int'] < 10)]['averageTra']*2


O3J.loc[O3J['local_time_int'] < 6, 'averageTra'] *= 0.1
O3J.loc[O3J['local_time_int'] >= 6].loc[O3J['local_time_int'] < 10, 'averageTra'] *= 0.1