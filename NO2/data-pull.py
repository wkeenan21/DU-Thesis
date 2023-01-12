import requests
import pandas as pd

# set variables
email = 'wkeenan21@gmail.com'
key = 'carmelswift52'
bdate = '20210401'
edate = '20211001'

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
coloradoData = requests.get(url = "https://aqs.epa.gov/data/api/dailyData/byState?email={}&key={}&param={}&bdate={}&edate={}&state=08".format(email,key,ozoneCode,bdate,edate))
coloradoData = coloradoData.json()

df = pd.DataFrame(coloradoData['Data'])

df.to_csv(r"C:\Users\willy\Downloads\Thesis\ozone\summer2021ozone.csv")