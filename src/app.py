

import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
import pickle

from prophet import Prophet
from prophet.plot import plot_plotly
from pmdarima.arima import auto_arima
from pylab import rcParams

# Read data

data_train_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_a = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-a.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)

data_train_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-train-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)
data_test_b = pd.read_csv('https://raw.githubusercontent.com/oreilly-mlsec/book-resources/master/chapter3/datasets/cpu-utilization/cpu-test-b.csv', parse_dates=[0], infer_datetime_format=True,index_col=0)


# Set time as index because we are working with time series

data_train_a.index = pd.to_datetime(data_train_a.index)
data_train_b.index = pd.to_datetime(data_train_b.index)


## Dataset A

# Use sarima model to fit the data

stepwise_model = auto_arima(data_train_a, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

# Fit the best model with df_a data

stepwise_model.fit(data_train_a)

# Save model

filename = '/workspace/Time-series/models/model_a.pkl'
pickle.dump(stepwise_model, open(filename,'wb'))


## Dataset B

# Use sarima model to fit the data

stepwise_model = auto_arima(data_train_b, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

# Fit the best model with df_a data

stepwise_model.fit(data_train_b)

filename = '../models/model_b.pkl'
pickle.dump(stepwise_model, open(filename,'wb'))