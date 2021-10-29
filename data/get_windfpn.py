import requests
from datetime import date, timedelta
import pandas as pd
import os

key = "hn1im51kb8yk3mf"

cdate = date(2018, 1, 1)
delta = timedelta(days=1)

#load wind farm names into list
farms = open("wind_farms.txt", "r")
listofwindfarms = []
temp = farms.readlines()
for line in temp:
    listofwindfarms.append(line.strip())

# collect 365 days of wind FPN data starting from 01/01/2018
for j in range(365):
    for i in range(48):
        res = open("windfpn.csv", "a")
        f = open("temp1.csv", "w")
        
        #get data from API and store in temp1.csv
        response = requests.get("https://api.bmreports.com/BMRS/PHYBMDATA/v1?APIKey="+key+"&SettlementDate="+str(cdate)+"&SettlementPeriod="+str(i+1)+"&ServiceType=csv")
        f.write(response.text)
        f.close()

        #delete temp2.csv if it exists
        os.remove("temp2.csv")

        #keep only first 2000 rows in f & add missing column headers
        with open('temp1.csv','r') as infile, open('temp2.csv','a') as outfile:
            infile.readline()
            outfile.write("1,2,3,4,a,b,c\n")
            for k in range(2000):
                outfile.write(infile.readline())
            infile.close()
            outfile.close()

        #api data to pandas dataframe object
        df = pd.read_csv("temp2.csv")

        #keep PN only
        df = df[df['1'] == 'PN']
        
        #keep generators only 
        for index, row in df.iterrows():
            if row['c'] <= 0:
                df = df.drop(index)

        #remove duplicates
        df = df.drop_duplicates(subset=['2'], keep='last')

        #keep wind farms only
        for index, row in df.iterrows():
            keep = False
            for farm in listofwindfarms:
                if farm in row['2']:
                    keep = True
            if keep == False:
                df = df.drop(index)

        #sum period wind fpn for each unit and write to windfpn.csv
        list = []
        for index, row in df.iterrows():
            list.append(row['c'])

        res.write(str(cdate)+"-"+str(i+1)+","+str(sum(list)) + "\n")

    cdate = cdate + delta
