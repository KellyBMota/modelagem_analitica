# -- coding: utf-8 --
"""
Created on Mon Oct  5 12:40:49 2020

@author: kelly
"""
from dateutil.parser import parse 
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

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

#HOLT-WINTERS
def holt_winters_forecast(history, config):
    is_exp, is_damped = config
    model = ExponentialSmoothing(history)
    model_fit = model.fit(optimized=True)
    yhat = model_fit.predict(len(history), len(history))
    return yhat

actual = df_test.value
history = [x for x in df_train.value]

config = [True, False]
predictions = list()
for i in range(len(actual)):
    yhat = holt_winters_forecast(history, config)
    predictions.append(yhat)
    obs = actual[i]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
rmse = sqrt(mean_squared_error(df_test, predictions))
print('rmse: %.3F', rmse)
plt.plot(actual)
plt.plot(predictions, color='red')
plt.show()