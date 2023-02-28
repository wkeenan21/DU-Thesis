import numpy
import pandas
import pandas as pd

denverWeather = pd.read_csv(r"C:\Users\willy\Documents\GitHub\DU-Thesis\Ozone\csvs\weather\Hourly2021\72466693067.csv")
denverWeather['DATE'] = pd.to_datetime(denverWeather['DATE'], format='%Y-%m-%dT%H:%M:%S')
julyW = denverWeather[(denverWeather['DATE'] > '2021-07-01') & (denverWeather['DATE'] < '2021-08-01')]
