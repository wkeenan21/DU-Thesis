import requests
import pandas as pd
import numpy as np
# from PIL import Image
# from osgeo import gdal
import requests
# set variables
email = 'wkeenan21@gmail.com'
key = 'carmelswift52'
bdateJuly = '20210701'
edateJuly = '20210731'
bdateAug = '20210801'
edateAug = '20210831'
bbox = "39.259770,-105.632996,40.172953,-104.237732"
bbox = bbox.split(',')
ozoneCode= '44201'
denverOzoneJuly = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,ozoneCode,bdateJuly,edateJuly, bbox[0], bbox[2], bbox[1], bbox[3])).json()


# find parameters
params = requests.get(url = "https://aqs.epa.gov/data/api/list/parametersByClass?email={}&key={}&pc=CRITERIA".format(email,key))
params = params.json()

# find states
states = requests.get(url = "https://aqs.epa.gov/data/api/list/states?email={}&key={}".format(email, key)).json()

# find ozone code
for i in params['Data']:
    print(i['value_represented'])
    if 'Ozone' in i['value_represented']:
        ozoneCode = (i['code'])
    elif 'Nitrogen' in i['value_represented']:
        NO2 = (i['code'])
    elif 'PM2.5' in i['value_represented']:
        PM25 = (i['code'])

denverOzoneJuly = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,ozoneCode,bdateJuly,edateJuly, bbox[0], bbox[2], bbox[1], bbox[3])).json()
denverOzoneAugust = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,ozoneCode,bdateAug,edateAug, bbox[0], bbox[2], bbox[1], bbox[3])).json()
denverNO2July = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,NO2,bdateJuly,edateJuly, bbox[0], bbox[2], bbox[1], bbox[3])).json()
denverNO2August = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,NO2,bdateAug,edateAug, bbox[0], bbox[2], bbox[1], bbox[3])).json()
denverPM25July = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,PM25,bdateJuly,edateJuly, bbox[0], bbox[2], bbox[1], bbox[3])).json()


OzoneDfJuly = pd.DataFrame(denverOzoneJuly['Data'])
OzoneDfAug = pd.DataFrame(denverOzoneAugust['Data'])
NO2DfJuly = pd.DataFrame(denverNO2July['Data'])
NO2DfAug = pd.DataFrame(denverNO2August['Data'])
PM25DfJuly = pd.DataFrame(denverPM25July['Data'])

OzoneDfJuly.to_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\JulyOzone.csv")
OzoneDfAug.to_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\today\AugOzone.csv")

NO2DfJuly.to_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\NO2\JulNO2.csv")
PM25DfJuly.to_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\PM25\JulPM25.csv")


dateRange = OzoneDfJuly['date_local'].unique()
times = OzoneDfJuly['time_local'].unique()
sitesOzone = OzoneDfJuly['site_number'].unique()
sitesNO2 = NO2DfJuly['site_number'].unique()

frames = [OzoneDfJuly, OzoneDfAug]
OzoneJulyAug = pd.concat(frames)

sitesNO2
sitesOzone
# days = {}
# for name in dateRange:
#     days[name] = pd.DataFrame(columns=OzoneDf.columns)
#
# for date in dateRange:
#     for time in times:
#         for index, row in denverDf.iterrows():
#             if row['date_local'] == date and row['time_local'] == time:
#                 days[date].loc[len(days[date].index)] = row

# for name in dateRange:
#     days[name].to_csv(r"C:\Users\willy\Downloads\Thesis\ozone\ozone{}.csv".format(name))


