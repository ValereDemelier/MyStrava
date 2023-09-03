#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 16:47:43 2021

@author: valeredemelier
Scraping strava API and Analyzing run data
"""
import requests
import urllib3
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from chord import Chord

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"

payload = {
    'client_id': "61095",
    'client_secret': 'SECRET',
    'refresh_token': 'TOKEN,
    'grant_type': "refresh_token",
    'f': 'json'
}

print("Requesting Token...\n")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']
print("Access Token = {}\n".format(access_token))

header = {'Authorization': 'Bearer ' + access_token}
param = {'per_page': 200, 'page': 1}
my_dataset = requests.get(activites_url, headers=header, params=param).json()

#importing both data sets including strava data and old nike data
strava_all_cols = json_normalize(my_dataset)
cols = ['name', 'distance','average_speed', 'moving_time', 'elapsed_time',
        'total_elevation_gain', 'id','achievement_count','start_date_local']
all_data_cols = ['date', 'km', 'elapsed_time']

# Read in old running data from the NIKE run app
nike = pd.read_csv('/Users/valeredemelier/Downloads/Running CSV.csv', parse_dates=True, nrows=32)



#Making both data sets the same so I can merge old data and new incoming data
#Create Datetime objects that show start time and date
strava = strava_all_cols[cols]
strava['km'] = strava['distance']/1000
strava ['start_date_local'] = pd.to_datetime(strava['start_date_local'])
strava['start_time'] = strava['start_date_local'].dt.time
strava ['date']= strava['start_date_local'].dt.date
strava['day'] = strava['start_date_local'].dt.weekday
strava['month'] = strava['start_date_local'].dt.month
y = strava['yr-month'] = strava['start_date_local'].dt.year.astype('str') + \
    '-' + strava['month'].astype('str')
strava['yr_month'] = pd.to_datetime(y)



#Making both data sets the same so I can merge old data and new incoming data
strava_small = strava[all_data_cols]
strava_small.columns = ['date', 'km', 'time']
strava_small['time'] = strava_small['time']/60
nike['Date'] = pd.to_datetime(nike['Date'])
nike.columns = ['date', 'km', 'time']
#Final Dataset combining both:
all_runs = nike.append(strava_small).sort_values('date')
# Drop long walk data point
all_runs = all_runs[all_runs.time <500]
strava = strava[strava.elapsed_time<5000]

#Plotting all runs in a line graph
def lineplot(x='date',y='km',df=all_runs):
    plt.style.use('seaborn-dark-palette')
    fig, ax = plt.subplots(figsize=(16,5))
    ax.plot(df[x], df[y], marker='o',linestyle='--')
    fig.autofmt_xdate(rotation=45)
    ax.set_xlabel('Date')
    ax.set_ylabel('Distance Ran (km)')
    ax.set_title("Valere's Running")
    ax.annotate('Stress Fracture', xy=('2020-05-16',21), 
            xytext=('2020-06-16',20), 
            arrowprops={'arrowstyle':'->','color':'red'})
    ax.annotate('First Half Marathon', xy=('2020-05-10',21), 
            xytext=('2020-03-15',15), 
            arrowprops={'arrowstyle':'->','color':'blue'})
    ax.annotate('Achillies Trouble', xy=('2020-10-29',8.5), 
            xytext=('2020-11-01',13), 
            arrowprops={'arrowstyle':'->','color':'red'})
    ax.annotate('Recovery', xy=('2020-12-23',2.23), xytext=('2020-12-07',5), 
            arrowprops={'arrowstyle':'->','color':'green'})
    plt.show()
    plt.close()
#Lineplot by Week beginning on Monday
def weekly_lineplot():
    ''' Define the weekly data set and plot is as a lineplot'''
    weekly_df = pd.DataFrame(all_runs.resample('W-SUN', on= 'date')['km'].sum())
    print(weekly_df)
    sns.set()
    fig, ax = plt.subplots(figsize=(16,5))
    _ = plt.plot(weekly_df.index, weekly_df['km'], marker='.',linestyle='--')
    _ = plt.fill_between(weekly_df.index, weekly_df.km, alpha=.2)
    _ = plt.xlabel('date')
    _ = plt.ylabel('Distance (km)')
    plt.show()
    plt.close()

#Look at distance run per month-year as a bar graph
month_year_order = ['Mar-20', 'Apr-20', 'May-20','Jun-20', 'Jul-20', 'Aug-20',
                                   'Sep-20','Oct-20','Nov-20',
                                   'Dec-20','Jan-21', 'Feb-21', 'Mar-21']

all_runs['date'] = pd.to_datetime(all_runs['date'])
by_month = pd.DataFrame(all_runs.resample('M', on='date')['km'].sum())
def labeled_barplot(x=by_month.index,y='km', df=by_month, xticks=month_year_order):
    fig, ax = plt.subplots(figsize=(12,7))
    sns.set_style('dark')
    splot = sns.barplot(x= x,y=y, data=df, 
                        palette='pastel')
    for p in splot.patches:
        splot.annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', xytext = (0, 10), 
                       textcoords = 'offset points')
    splot.set_xticklabels(xticks)
    splot.set_xlabel('Month of the Year')
    
#Cross tab of run count by day and month
day_order = ["Monday", "Tuesday", "Wednesday", 
             "Thursday", "Friday", "Saturday", "Sunday"]
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
all_runs['day_name'] = all_runs['date'].dt.weekday
all_runs['month'] = all_runs['date'].dt.month
def crosstab_running(var1='month', var2='day_name', df=all_runs, yticklabel=day_order, xticklabel=month_order, cmap='BuGn'):
    fig, ax = plt.subplots()
    crosstab= pd.crosstab(df[var2], df[var1])
    print(crosstab)
    sns.heatmap(crosstab, annot=True, linewidth=.1, yticklabels=yticklabel, 
            xticklabels=xticklabel, cmap = 'BuGn' )
    plt.show()
    plt.close()


def ecdf(data):
    n = len(data)
    x = np.sort(data)
    y = np.arange(1, n+1)/n
    return x, y
sns.set()
x_km, y_km = ecdf(all_runs.km)
_=plt.plot(x_km, y_km, marker='.', linestyle='none')
_=plt.xlabel('Distance run (km)')
_ = plt.ylabel('ECDF')
plt.show()
plt.clf()

mapping = {}
for i in range(22):
    if i <= 5 :
        mapping[i] = 'short run'
    elif i < 10:
        mapping[i] = 'medium run'
    elif i >=10:
        mapping[i] = 'long run'
    elif i >=15:
        mapping[i] = 'very long run'
all_runs['category'] = np.round(all_runs.km).map(mapping)


_ = sns.swarmplot(x='category', y='time', data=all_runs)
_ = plt.xlabel('Run distance')
_ = plt.ylabel('Elapsed time')
plt.show()
plt.clf()


strava['km'] = strava.distance/1000
strava['category'] = np.round(strava.km).map(mapping)
_ = sns.boxplot(x='category', y='average_speed', data=strava)
_ = plt.xlabel('Type of Run')
_ = plt.ylabel('Average Speed')
plt.show()
plt.clf()


















