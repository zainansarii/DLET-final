from pandas.core.arrays import string_
import requests
import pandas as pd
import datetime as dt

key = "hn1im51kb8yk3mf"
time = dt.datetime(2018, 1, 1, 00, 00, 00)
timedelta = dt.timedelta(minutes=30)

# get 1 year of rolling system demand data from 01/01/2018
for j in range(365):
  for i in range(48):
      response = requests.get("https://api.bmreports.com/BMRS/ROLSYSDEM/v1?APIKey=" + key + "&FromDateTime=" + str(time) + "&ToDateTime=" + str(time) + "&ServiceType=csv")
      time = time + timedelta
      saveFile = open('temp1.csv', 'a')
      saveFile.write(response.text)
      saveFile.close()

# clean and store data
df = pd.read_csv("temp1.csv")
df = df.loc[df.index.str.contains('VD')]
df['ROLLING SYSTEM DEMAND'].to_csv('rsd.csv', index=False)
