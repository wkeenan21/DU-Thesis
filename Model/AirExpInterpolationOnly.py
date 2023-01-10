import arcpy
from arcpy import env
env.overwriteOutput = True
import pandas as pd
import arcpy.sa
import datetime
from datetime import datetime,timedelta
import os
import numpy as np
env.workspace= os.getcwd()
env.extent =  arcpy.Extent(-105.05, 39.6, -104.82, 39.8)

def ReadDataStation(datafile = "PM25\EPA\epa_01_02_2021.xlsx", hour="", pollutants ="PM25", shp ="", excludelist=[]):
    # stationID= ""
    # "Station"+stationID
    # df_stationinfo['Date'] = pd.to_datetime(df_stationinfo['Date'], format="%m_%d_%Y")
    f = pd.ExcelFile(datafile)
    sheet_names = f.sheet_names
    stationinfo = {}
    for sheet_name in sheet_names:
        id =  int(sheet_name)
        sheet = f.parse(sheet_name = sheet_name)
        ##select pm 2.5 data for an hour
        pd_select = sheet.loc[sheet['Time'] == hour]
        if len(pd_select.index)>0:
            pollutantvalue =  pd_select.iloc[-1]["PM25"]
            stationinfo[id] = pollutantvalue

    ##add a field to the shapefile if not exist
    # Set local variables
    inFeatures =shp
    fieldName1 = pollutants
    # Execute AddField
    arcpy.AddField_management(inFeatures, fieldName1, "DOUBLE")
    ##use updatecursor to update info
    cursor = arcpy.da.UpdateCursor(shp, fields)
    for row in cursor:
        if row[0] in stationinfo.keys():
            row[1] = stationinfo[row[0]]
        else:
            row[1] = -1
        if row[0] in excludelist:
            row[1] = -1
        if row[1] >=0:
            row[1] = row[1]
        else:
            row[1] = -1
        print(row)
        cursor.updateRow(row)
    del row
    del cursor
    return stationinfo

##todo: run average ...
def RunAverage():
    return 0

def RunInterpolation(days_total = 30, days_start = "01_02_2021", fields = ["Site_ID","PM25"],
                     stationlist =[155, 167, 168, 175]):
    file_by_month = []
    interpotype = ["LMA"]
    ##todo: modify
    shplist = ["PM25\SHP\EPAStations_Denver_C.shp", "PM25\SHP\LMASensors_Denver_Correct_New.shp","PM25\SHP\extract.shp"]

    cnt_station =  len(stationlist)
    valuelist = {}
    for i_station in stationlist:
        valuelist[i_station] = np.zeros(24 * 3).reshape(24, 3)

    outfilelist = []
    monthlyraster = {}
    monthly_average={}
    for i_month in range(10):
        monthlyraster[i_month+1] = []
        monthly_average[i_month+1] =0
    for i_day in range(days_total):
        datetime_object = datetime.strptime(days_start, "%m_%d_%Y")
        newdate = datetime_object + timedelta(days=i_day)
        dateinfostr = newdate.strftime(format="%m_%d_%Y")
        for i_type in range(1):
            ##todo: compute daily average
            for i_hour in range(24):
                shp = ""
                if interpotype[i_type]=="EPA":
                    filename = "PM25\EPA\epa_"+dateinfostr+".xlsx"
                    shp = shplist[0]
                    ReadDataStation(filename, i_hour, shp=shp)
                else: ##todo: modify
                    filename =  "PM25\LMACorrected\lma_" + dateinfostr + ".xlsx"
                    shp =shplist[1]
                    stationinfocomplete =ReadDataStation(filename, i_hour, shp=shp, excludelist=stationlist)
                ##exclude points with missing values

                positiveshp =arcpy.management.SelectLayerByAttribute(shp,"NEW_SELECTION",'"PM25" >= 0' )


                # outraster = arcpy.sa.Kriging(positiveshp, fields[1], "Spherical 0.000515 # # #", cell_size="5.15484000000015E-04",
                #                  search_radius="VARIABLE 12",
                #                  out_variance_prediction_raster="")

                outraster = arcpy.sa.Idw(positiveshp, z_field=fields[1],cell_size="5.15484000000015E-04",)
                outraster.save("Interpolation/raster_"+dateinfostr+"H"+str(i_hour)+".tif")
                # outraster = arcpy.sa.NaturalNeighbor(positiveshp, z_field=fields[1])
                outraster = arcpy.Raster("Interpolation/raster_"+dateinfostr+"H"+str(i_hour)+".tif")
                rasterfilename = "Interpolation/raster_"+dateinfostr+"H"+str(i_hour)+".tif"
                if i_hour ==0:
                    dailyraster = outraster
                else:
                    dailyraster = dailyraster + outraster
                outraster.save(rasterfilename)
            print(i_hour)
            dailyraster= dailyraster / (i_hour+1)
            dailyraster.save("Interpolation/raster_"+dateinfostr+".tif")
        # monthlyraster[newdate.month].append(dailyraster)

def ComputeMonthlyAvg(days_start, days_total, month_use = 1):
    monthlyfile_list = {}
    monthlyraster={}
    monthly_average={}
    for i_month in range(9):
        monthlyfile_list[i_month+1] =[]
        monthlyraster[i_month+1] =[]
        monthly_average[i_month+1] =[]

    for i_day in range(days_total):
        datetime_object = datetime.strptime(days_start, "%m_%d_%Y")
        newdate = datetime_object + timedelta(days=i_day)
        dateinfostr = newdate.strftime(format="%m_%d_%Y")
        monthlyfile_list[datetime_object.month].append("Interpolation/raster_"+dateinfostr+".tif")

    for i_file in monthlyfile_list[month_use]:
        monthlyraster[month_use] .append( arcpy.Raster(i_file))
    for i_raster in range(len(monthlyraster[month_use])):
        if i_raster ==0:
            monthly_average[month_use] =monthlyraster[month_use][i_raster]
        else:
            monthly_average[month_use] =  monthly_average[month_use] + monthlyraster[month_use][i_raster]
    monthly_average[month_use] =  monthly_average[month_use]/len(monthlyraster[month_use])
    monthly_average[month_use].save("Interpolation/raster_M_"+str(month_use)+".tif")
    # return outfilelist


import sys
arcpy.CheckOutExtension("spatial")

if len(sys.argv) > 1:
    i_month = int(sys.argv[1])  # [50,100,200]
else:
    i_month = 8

days_start = "0"+str(i_month+1)+"_01_2021"
fields = ["Site_ID","PM25"]
stationlist = []
# RunInterpolation(32,days_start,fields,stationlist)

days_start = "0"+str(i_month+1)+"_24_2021"
ComputeMonthlyAvg(days_start, days_total=15, month_use = i_month+1) ##27 for sep, if use the real start prediction date
##todo: pick sites;  pick time; analyze difference
arcpy.CheckInExtension("spatial")