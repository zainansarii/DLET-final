import requests
from datetime import date, timedelta
import pandas as pd
import os

key = "hn1im51kb8yk3mf"

cdate = date(2018, 1, 1)
delta = timedelta(days=1)

# collect 365 days of FPN data starting from 01/01/2018
for j in range(365):
    for i in range(48):
        res = open("fpn.csv", "a")
        f = open("temp1.csv", "w")

        #get data from API and store in temp1.csv
        response = requests.get("https://api.bmreports.com/BMRS/PHYBMDATA/v1?APIKey="+key+"&SettlementDate="+str(cdate)+"&SettlementPeriod="+str(i+1)+"&ServiceType=csv")
        f.write(response.text)
        f.close()

        #delete temp2.csv if it exists
        os.remove("temp2.csv")

        #keep only first 2000 rows in temp1.csv, add missing column headers, and store result in temp2.csv
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
                
        #sum period fpn for each unit and write to fpn.csv
        period_fpn = []
        for index, row in df.iterrows():
            period_fpn.append(row['c'])

        res.write(str(cdate)+"-"+str(i+1)+","+str(sum(period_fpn)) + "\n")

    cdate = cdate + delta
