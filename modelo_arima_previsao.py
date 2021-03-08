# -*- coding: utf-8 -*-
"""
@author: kelly
"""

import matplotlib.pyplot as plt
import datetime as dt
from matplotlib.pylab import rcParams

import pandas as pd
from matplotlib import pyplot

rcParams['figure.figsize'] = 15, 6
plt.rcParams.update({'figure.figsize': (10, 7), 'figure.dpi': 120})

dateparse = lambda dates: dt.datetime.strptime(dates, "%Y-%m-%dT%H:%M:%S.%f%z")

# LEITURA DO ARQUIVO DE TREINO
df_train = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_train.csv', parse_dates=['datetime'], date_parser=dateparse, index_col=1)

# LEITURA DO ARQUIVO DE TESTE
df_test = pd.read_csv('ElectricDemandForecasting-DL-master_data_hourly_20140102_20191101_test.csv', parse_dates=['datetime'], date_parser=dateparse, index_col=1)

# convert to datetime
df_test['datetime'] = pd.to_datetime(df_test['datetime'], utc=True)
df_train['datetime'] = pd.to_datetime(df_train['datetime'], utc=True)

df_test = df_test.set_index('datetime')
df_train = df_train.set_index('datetime')

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(df_test)
pyplot.show()

from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA

model = ARIMA(df_test, order=(1,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())
# plot residual errors
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())


from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

train, test = df_test.value, df_train.value
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('pos=%f, predicted=%f, expected=%f' % (t, yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()