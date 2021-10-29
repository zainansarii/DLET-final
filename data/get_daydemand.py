import requests
from datetime import date, timedelta
import pandas as pd

key = "hn1im51kb8yk3mf"

# get 3 months of day demand data from 01/01/2018
response = requests.get("https://api.bmreports.com/BMRS/FORDAYDEM/v1?APIKey="+key+"&FromDate=2018-01-01&ToDate=2018-03-31&ServiceType=CSV")
f = open("temp1.csv", "w")
f.write(response.text)

# clean and store data
df = pd.read_csv("temp1.csv")
df['FORECAST DAY AND DAY AHEAD DEMAND DATA'].to_csv("daydemand.csv", index=False)
