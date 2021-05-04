# -- coding: utf-8 --
"""
Created on Mon Oct  5 12:40:49 2020

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
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing

rcParams['figure.figsize'] = 15, 6
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

dateparse = lambda dates: dt.datetime.strptime(dates, "%Y-%m-%dT%H:%M:%S.%f%z")

# LEITURA DO ARQUIVO DE TREINO
df_train = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_train.csv', parse_dates=['datetime'], date_parser=dateparse)

# LEITURA DO ARQUIVO DE TESTE
df_test = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_test.csv', parse_dates=['datetime'], date_parser=dateparse)

# convert to datetime
df_test['datetime'] = pd.to_datetime(df_test['datetime'], utc=True)
df_train['datetime'] = pd.to_datetime(df_train['datetime'], utc=True)

def plot_df_train(df_train, x, y, title="", xlabel='datetime', ylabel='value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

# GRÁFICO INICIAL
plot_df_train(df_train, x=df_train.datetime, y=df_train.value, title='Eletric Demand Forecasting')
    
# MODELO DE SUAVIZAÇÃO EXPONENCIAL DE PREVISÃO
dd= np.asarray(df_train.value)
y_hat = df_test.copy()

# PEGA O ULTIMO VALOR
y_hat['ES'] = dd[len(dd)-1]
alpha = 0.3
for i in range(1,len(y_hat)):
    y_hat.ES[i] = alpha * y_hat.ES[i] + (1 - alpha) * y_hat.value[i-1]

# PLOTANDO RESULTADO
plt.figure(figsize=(12,8))
plt.plot(df_train.datetime, df_train['value'], label='Train')
plt.plot(df_test.datetime,df_test['value'], label='Test')
plt.plot(y_hat.datetime,y_hat['ES'], label='Exponencial Smoothing Forecast')
plt.legend(loc='best')
plt.title("Exponencial Smoothing Forecast")
plt.show()