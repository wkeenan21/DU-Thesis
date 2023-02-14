#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import matplotlib.dates as md 
from matplotlib.pyplot import figure
import matplotlib.patches as patches

#load in data
AllData = pd.read_csv(r"C:\Users\willy\Desktop\LotteryStats.csv", parse_dates=['Date'])

#Delete useless column
AllData.drop('Number', axis=1, inplace=True)

#Add column for day of week
AllData['DOW'] = AllData['Date'].dt.day_name()

#find dates where two trips were launched
Duplicates = AllData.duplicated()

#Add column to denote days with two trips
Trips = [1] * len(Duplicates)
AllData = AllData.assign(Trips = Trips)

#Delete double rows but add 1 to the trip column. Divide by two to represent relative number of trips
for i in range(len(Duplicates)):
    if Duplicates[i] == True:
        AllData.drop(index = i)
        AllData.loc[i-1, 'Trips'] = AllData['Trips'][i-1] + 1
        AllData.loc[i-1, 'Total'] = AllData['Total'][i-1]/2

#Add year as a new column
Year = []
for i in range(len(AllData['Date'])): Year.append(AllData['Date'].loc[i].year)
AllData = AllData.assign(Year = Year)

#Create column for dates without years
DM = []
for i in range(len(AllData['Date'])): DM.append('2020-' + str(AllData['Date'].loc[i].month) + '-' + str(AllData['Date'].loc[i].day))
AllData = AllData.assign(Date2 = DM)
AllData['Date2'] = pd.to_datetime(AllData['Date2'])

#Add column converting Total to Probability
prob = []
for i in AllData['Total']:
    if i == 0:
        prob.append(1)
    else:
        prob.append(500/i)
AllData = AllData.assign(Probability = prob)

#create small trips
SmallTrips = AllData[AllData['Size'] == 'Small']
SmallTrips = SmallTrips.reset_index(drop=True)

#create standard trips list
StandardTrips = AllData[AllData['Size'] == 'Standard']
StandardTrips = StandardTrips.reset_index(drop=True)

Winter = pd.DataFrame(columns=list(AllData.columns.values))
for i in range(len(AllData['Date'])):
    if AllData['Date'][i].month > 10 or AllData['Date'][i].month < 3:
        Winter = Winter.append(AllData.loc[i])
Winter = Winter.reset_index(drop=True)

#Viewing the Data
#Create functions for plotting data:
def plotLine(df, xaxis, yaxis, filt = None, yaxis2 = None):
    fig, ax = plt.subplots()

    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.plot(df[filter][xaxis], df[filter][yaxis], label = year)
        ax.legend(frameon = True)
    else:
        ax.plot(df[xaxis], df[yaxis])

    if yaxis2 != None:
        ax2 = ax.twinx()
        ax2.plot(xaxis, yaxis2, color='red')
        ax2.set_ylabel(yaxis2)

    ax.xaxis.set_major_locator(md.DayLocator(interval = 15))
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.show()

def plotScatter(df, xaxis, yaxis, filt = None, yaxis2 = None):
    fig, ax = plt.subplots()

    if filt != None:
        for year in df[filt].unique():
            filter = df[filt] == year
            ax.scatter(df[filter][xaxis], df[filter][yaxis], label = year)
        ax.legend(frameon = True)
    else:
        ax.scatter(df[xaxis], df[yaxis])

    if yaxis2 != None:
        ax2 = ax.twinx()
        ax2.scatter(xaxis, yaxis2, color='red')
        ax2.set_ylabel(yaxis2)

    ax.xaxis.set_major_locator(md.DayLocator(interval = 15))
    ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation = 90)

    ax.set_xlabel(xaxis)
    ax.set_ylabel(yaxis)
    fig = plt.gcf()
    fig.set_size_inches(12, 7)

    plt.show()

#This function averages the probability's by date and creates a new data frame we can use to plot the data.
def FindProbability(df):
    Days = df['Date2'].unique()
    Days = list(Days)
    Days.sort()
    DOY = []
    avgChances = []
    for i in range(len(Days)):
        name = Days[i]
        day = df[df['Date2'] == Days[i]]
        var = sum(day['Total'])
        count = len(day.index)
        average = var/count
        DOY.append(name)
        avgChances.append(average)

    proba = np.array(avgChances)
    proba = 100/(proba/5)
    proba = proba.ravel()
    lists = [DOY, proba, avgChances]

    Probability = pd.DataFrame(lists).transpose()
    Probability.columns = ['Date', 'Probability', 'Average Chances']
    return Probability

print('Standard Trips, Total vs Date')
plotLine(Winter, 'Date2', 'Total', 'Year')

winterProba = FindProbability(Winter)

print('Standard Trips')
plotLine(winterProba, 'Date', 'Probability')