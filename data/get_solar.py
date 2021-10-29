import os
import requests
import pandas as pd
from datetime import date, timedelta

key = "hn1im51kb8yk3mf"

sdate = date(2018, 1, 1)
delta = timedelta(days=1)

# get 1 year of day-ahead solar forecasts for each settlement period from 01/01/2018
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

    # filter day-ahead solar data and store
    df_solar = df.loc[df['Power System Resource  Type'].str.contains('Solar') & df['Process Type'].str.contains('Day ahead')]
    df_solar = df_solar[['Settlement Date', 'Settlement Period', 'Quantity']]
    df_solar = df_solar.iloc[::-1]
    df_solar['Quantity'].to_csv('solar.csv', mode='a', header=False, index=False)

    os.remove('temp1.csv')
