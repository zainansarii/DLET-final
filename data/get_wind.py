import os
import requests
import pandas as pd
from datetime import date, timedelta

key = "hn1im51kb8yk3mf"

sdate = date(2018, 1, 1)
delta = timedelta(days=1)

# get 1 year of day-ahead wind forecasts for each settlement period from 01/01/2018
for i in range(365):
    day = sdate + timedelta(days=i)
    dayString = "20" + day.strftime("%y-%m-%d")
    response = requests.get("https://api.bmreports.com/BMRS/B1440/v1?APIKey=" + key + "&SettlementDate=" + dayString + "&Period=*&ServiceType=csv")
    saveFile = open('temp1.csv', 'a')
    saveFile.write(response.text)
    saveFile.close()

    # clean data from API
    with open('temp1.csv', 'r') as infile:
        data = infile.read().splitlines(True)
    with open('temp1.csv', 'w') as outfile:
        outfile.writelines(data[4:])

    df = pd.read_csv("temp1.csv")
    df = df[df['*Time Series ID'].str.contains("NGET")]

    # filter day-ahead wind data
    df_wind = df.loc[df['Power System Resource  Type'].str.contains('Wind') & df['Process Type'].str.contains('Day ahead')]
    df_wind = df_wind[['Power System Resource  Type', 'Settlement Date', 'Settlement Period', 'Quantity']]

    # wind forecast = wind-offshore forecast + wind-onshore forecast
    df_wind['final'] = df_wind['Quantity'] + df_wind['Quantity'].shift(-1)
    df_wind = df_wind.iloc[::2, :]
    
    # store data in csv
    df_wind = df_wind.iloc[::-1]
    df_wind['final'].to_csv('wind.csv', mode='a', header=False, index=False)

    os.remove('temp1.csv')
