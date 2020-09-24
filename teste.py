# -- coding: utf-8 --
"""
Created on Sat Sep 12 12:23:09 2020

@author: kelly
"""
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

dateparse = lambda dates: dt.datetime.strptime(dates, "%Y-%m-%dT%H:%M:%S.%f%z")

# Import as Dataframe
df = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_train.csv', parse_dates=['datetime'], index_col='datetime', date_parser=dateparse)

#reset
df.reset_index(inplace=True)

# convert to datetime
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

# # perform GroupBy operation over monthly frequency
table_per_year = df.set_index('datetime').groupby(pd.Grouper(freq='Y'))['value'].mean().reset_index()
table_per_month = df.set_index('datetime').groupby(pd.Grouper(freq='M'))['value'].mean().reset_index()

#unic years
table_per_year['year'] = [d.year for d in table_per_year.datetime]
table_per_month['year'] = [d.year for d in table_per_month.datetime]
years = table_per_year['year'].unique()

# Prep Colors
np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'value', data=table_per_month.loc[table_per_month.year==y, :], color=mycolors[i], label=y)
        plt.text(table_per_month.loc[table_per_month.year==y, :].shape[0]-.9, table_per_month.loc[table_per_month.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)
plt.show()