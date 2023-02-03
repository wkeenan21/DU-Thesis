import requests
import pandas as pd
import numpy as np

# set variables
email = 'wkeenan21@gmail.com'
key = 'carmelswift52'
bdate = '20210501'
edate = '20210701'
bbox = "39.259770,-105.632996,40.172953,-104.237732"
bbox = bbox.split(',')

# find parameters
params = requests.get(url = "https://aqs.epa.gov/data/api/list/parametersByClass?email={}&key={}&pc=AQI POLLUTANTS".format(email,key))
params = params.json()

# find states
states = requests.get(url = "https://aqs.epa.gov/data/api/list/states?email={}&key={}".format(email, key)).json()

# find ozone code
for i in params['Data']:
    if 'Ozone' in i['value_represented']:
        ozoneCode = (i['code'])

# make request

# daily summary data
#coloradoData = requests.get(url = "https://aqs.epa.gov/data/api/dailyData/byState?email={}&key={}&param={}&bdate={}&edate={}&state=08".format(email,key,ozoneCode,bdate,edate))
#coloradoData = coloradoData.json()
#df = pd.DataFrame(coloradoData['Data'])
#df.to_csv(r"C:\Users\willy\Downloads\Thesis\ozone\summer2021ozone.csv")

denverBBOX = requests.get("https://aqs.epa.gov/data/api/sampleData/byBox?email={}&key={}&param={}&bdate={}&edate={}&minlat={}&maxlat={}&minlon={}&maxlon={}".format(email,key,ozoneCode,bdate,edate, bbox[0], bbox[2], bbox[1], bbox[3])).json()

denverDf = pd.DataFrame(denverBBOX['Data'])

dateRange = denverDf['date_local'].unique()
times = denverDf['time_local'].unique()
sites = denverDf['site_number'].unique()

days = {}
for name in dateRange:
    days[name] = pd.DataFrame(columns=denverDf.columns)

for date in dateRange:
    print(date)
    for time in times:
        for index, row in denverDf.iterrows():
            if row['date_local'] == date and row['time_local'] == time:
                days[date].loc[len(days[date].index)] = row


for name in dateRange:
    days[name].to_csv(r"C:\Users\willy\Downloads\Thesis\ozone\ozone{}.csv".format(name))

# define input data
data = np.asarray([0, 0, 0, 1, 1, 0, 0, 0])
data = data.reshape(1, 8, 1)

data