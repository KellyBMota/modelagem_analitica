# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 09:05:04 2020

@author: kelly
"""

from dateutil.parser import parse 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime as dt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6

dateparse = lambda dates: dt.datetime.strptime(dates, "%Y-%m-%dT%H:%M:%S.%f%z")

# Import as Dataframe
df = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_train.csv', parse_dates=['datetime'], index_col='datetime', date_parser=dateparse)

#reset
df.reset_index(inplace=True)

# convert to datetime
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

df_year = df
df_year['datetime'] = df['datetime']
df_year = df.set_index('datetime', drop=False).groupby([pd.Grouper(key='datetime',freq='Y')])['value'].mean().reset_index()

df_year = df_year.set_index('datetime')

plt.plot(df_year)