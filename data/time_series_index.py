import pandas as pd
from datetime import date, timedelta

#initialise start date, end date, and time-delta
sdate = date(2018, 1, 1)
edate = date(2020, 2, 28)
delta = edate - sdate

# add indexing column to list
data_index = []
for i in range(delta.days + 1):
    date = sdate + timedelta(days=i)
    dateString = date.strftime("%y%m%d")
    for j in range(48):
        data_index.append(dateString + f"{j + 1:02d}")

# add month column to list
data_month = []
for i in range(delta.days + 1):
    day = sdate + timedelta(days=i)
    monthString = day.strftime("%m")
    for j in range(48):
        data_month.append(monthString)

# add week column to list
data_week = []
count = 0
for i in range(delta.days + 1):
    for j in range(48):
        data_week.append(str(count%7 + 1)) 
    count += 1

# add settlement period column to list
data_sp = []
for i in range(delta.days + 1):
    for j in range(48):
        data_sp.append(str(j+1))

#create pandas dataframe from lists and export as CSV
df = pd.DataFrame(list(zip(data_index, data_month, data_week, data_sp)),
                                  columns =['id', 'month', 'day of week', 'settlement period'])

df.to_csv("column1-4.csv", index=False)
