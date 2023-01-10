import requests

# set variables
email = 'wkeenan21@gmail.com'
key = 'carmelswift52'
bdate = '20220101'
edate = '20220701'

# find parameters
params = requests.get(url = "https://aqs.epa.gov/data/api/list/parametersByClass?email={}&key={}&pc=AQI POLLUTANTS".format(email,key))
params = params.json()

# find states
states = requests.get(url = "https://aqs.epa.gov/data/api/list/states?email={}&key={}".format(email, key)).json()
states['Data']

for i in params['Data']:
    if 'Ozone' in i['value_represented']:
        ozoneCode = (i['code'])


coloradoData = requests.get(url = "https://aqs.epa.gov/data/api/dailyData/byState?email={}&key={}&param={}&bdate={}&edate={}&state=08".format(email,key,ozoneCode,bdate,edate))
coloradoData = coloradoData.json()
coloradoData['Data']