import requests
from datetime import date, timedelta
import pandas as pd

key = "hn1im51kb8yk3mf"

cdate = date(2018, 1, 1)
delta = timedelta(days=1)

# store column headers in file
f=open("temp1.csv", "a")
f.write("type, code, date, sp, price, volume\n")
f.close()

# collect 365 days of MIP data starting from 01/01/2018
for i in range(365):
    f = open("temp2.csv","w")
    response = requests.get("https://api.bmreports.com/BMRS/MID/v1?APIKey="+key+"&FromSettlementDate="+str(cdate)+"&ToSettlementDate="+str(cdate)+"&Period=*&ServiceType=CSV")
    f.write(response.text)
    f.close()
    with open("temp2.csv","r") as infile, open("temp1.csv", "a") as outfile:
        infile.readline()
        for i in range(48):
            outfile.write(infile.readline())
    cdate = cdate + delta

# clean and store data
df = pd.read_csv("temp1.csv")
df = df[~df[' code'].str.contains("N2EXMIDP")]
df[' price'].to_csv("mip.csv", index=False)
