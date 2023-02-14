fn = r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\gisData\TROPOMI\den.nc\S5P_NRTI_L2__O3_____20230130T204306_20230130T204806_27459_03_020401_20230130T212151\S5P_NRTI_L2__O3_____20230130T204306_20230130T204806_27459_03_020401_20230130T212151.nc"
O3 = nc.Dataset(fn)

# extract lat long and ozone data
vars = O3.groups['PRODUCT'].variables
ozone = vars['ozone_total_vertical_column'][0]
lats = vars['latitude'][0]
lons = vars['longitude'][0]

def conversionCode():
    pass
    # select area over denver, first must convert 4326 to 3857
    # def ConvertLon(lon):
    #     lonInEPSG3857 = (lon * 20037508.34 / 180)
    #
    #     return lonInEPSG3857
    # def ConvertLat(lat):
    #     latInEPSG3857 = (math.log(math.tan((90 + lat) * math.pi / 360)) / (math.pi / 180)) * (20037508.34 / 180)
    #     return latInEPSG3857

    # vConvertLat = np.vectorize(ConvertLat)
    # vConvertLon = np.vectorize(ConvertLon)
    # lons3857 = vConvertLon(lons)
    # lats3857 = vConvertLat(lats)

clippedLats = np.logical_and(lats > miny, lats < maxy)
clippedLons = np.logical_and(lons > minx, lons < maxx)
clipped = clippedLons*clippedLats

clipLats = clipped * lats
ozoneClipped = ozone * clipped

data = clipLats[~np.all(clipLats == 0, axis=0)]
idx = np.argwhere(np.all(data[..., :] == 0, axis=0))
a2 = np.delete(data, idx, axis=1)
