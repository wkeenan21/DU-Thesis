import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

June15 = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\Summer2021\ozone2021-06-15.csv")

sites = list(June15['site_number'].unique())

siteDfs = {}
for i in sites:
    siteDfs[str(i)] = June15[June15['site_number'] == i]

# for i in sites:
#     plt.plot(siteDfs[str(i)]['time_local'], siteDfs[str(i)]['sample_measurement'])

plt.plot(siteDfs[str('26')]['time_local'], siteDfs[str('26')]['sample_measurement'])
plt.title('June 15 2021: Ozone Levels in Downtown Denver')
plt.xlabel('Time')
plt.ylabel('Ozone concentration in PPM')
plt.show()
plt.close()

email = 'wkeenan21@gmail.com'
key = 'carmelswift52'
import requests
url = 'https://aqs.epa.gov/data/api/dailyData/bySite?email={}&key={}&param=44201&bdate=20210101&edate=20211231&state=08&county=031&site=0002'.format(email, key)
ozoneYear = requests.get(url).json()

ozoneYear = pd.DataFrame(ozoneYear['Data'])

ozoneYear['date_local'] = pd.to_datetime(ozoneYear['date_local'])

plt.plot(ozoneYear['date_local'], ozoneYear['first_max_value'])
plt.title('Ozone 1 Day Maximums: 2021')
plt.xlabel('Date')
plt.ylabel('Ozone concentration in PPM')
plt.show()