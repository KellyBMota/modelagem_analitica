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
from matplotlib.pylab import rcParamsddasdasd
from statsmodels.tsa.seasonal import seasonal_decompose
rcParams['figure.figsize'] = 15, 6
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

dateparse = lambda dates: dt.datetime.strptime(dates, "%Y-%m-%dT%H:%M:%S.%f%z")

# Import as Dataframe
df = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_train.csv', parse_dates=['datetime'], index_col='datetime', date_parser=dateparse)

def plot_df(df, x, y, title="", xlabel='datetime', ylabel='value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

#GRÁFICO INICIAL
plot_df(df, x=df.index, y=df.value, title='Eletric Demand Forecasting')

#reset
df.reset_index(inplace=True)

# convert to datetime
df['datetime'] = pd.to_datetime(df['datetime'], utc=True)

#PER YEAR
df_year = df
df_year['datetime'] = df['datetime']
df_year = df.set_index('datetime', drop=False).groupby([pd.Grouper(key='datetime',freq='Y')])['value'].mean().reset_index()

df_year = df_year.set_index('datetime')
plt.plot(df_year)
plt.title('PER YEAR')
plt.show()

#PER MONTH
df_month = df
df_month['datetime'] = df['datetime']
df_month = df.set_index('datetime', drop=False).groupby([pd.Grouper(key='datetime',freq='M')])['value'].mean().reset_index()

df_month = df_month.set_index('datetime')
plt.plot(df_month)
plt.title('PER MONTH')
plt.show()

#PER WEEK
df_week = df
df_week['datetime'] = df['datetime']
df_week = df.set_index('datetime', drop=False).groupby([pd.Grouper(key='datetime',freq='W')])['value'].mean().reset_index()

df_week = df_week.set_index('datetime')
plt.plot(df_week)
plt.title('PER WEEK')
plt.show()
df_week = df_week.reset_index()


#USANDO OS DADOS POR SEMANA PARA FACILITAR A VISUALIZAÇÃO
df = df_week

#HISTOGRAMA
plt.gca().set(title='Histograma', xlabel='valores', ylabel='Frequencia')
x=df.value
plt.hist(x, bins=50)


#BOX PLOT
df['year'] = [d.strftime('%Y') for d in df.datetime]
df['month'] = [d.strftime('%m') for d in df.datetime]
years = df['year'].unique()

fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='value', data=df, ax=axes[0])
sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([2014, 2018]), :])

axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
plt.show()


#TESTE ESTACIONARIDADE
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=10).mean()
    rolstd = pd.Series(timeseries).rolling(window=10).std()

    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

ts = df['value'] 
test_stationarity(ts)

# transformação log
ts_log = np.log(ts)
#ts_log = ts
plt.plot(ts_log)
plt.title('transformação log')
plt.show()

# #DECOMPOSE LOG

decomposition = seasonal_decompose(ts_log, period=14) 
trend = decomposition.trend 
seasonal = decomposition.seasonal 
residual = decomposition.resid 
plt.subplot(411) 
plt.plot(ts_log, label='Original') 
plt.legend(loc='best') 
plt.subplot(412) 
plt.plot(trend, label='Trend') 
plt.legend(loc='best') 
plt.subplot(413) 
plt.plot(seasonal,label='Seasonality') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.plot(residual, label='Residuals') 
plt.legend(loc='best') 
plt.subplot(414) 
plt.tight_layout()
plt.show() 

ts_log_decompose = residual 
ts_log_decompose.dropna(inplace=True) 
test_stationarity(ts_log_decompose)
