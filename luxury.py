import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression as lr
# from end_data import end_value

# 70000 seats total
# Upper: 25000
# Middle: 20000
# Lower: 20000
# Luxury: 5000

df = pd.read_csv('data/LuxuryAttendance.csv')

# df.rename(columns={c: c.strip() for c in df.columns.values.tolist()}, inplace=True) 
def end_value(data, level):
    data = data.reset_index()
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])
    data = data.dropna()

    data = data.astype({' Home Win %': 'float64', ' Visitor Win %': 'float64', ' Home Market Index': 'float64', ' Home Stars Index': 'int64', ' Visitor Stars Index': 'int64', ' Home AFP': 'int64', ' Visitor AFP': 'int64', ' Ticket Price': 'float64', ' Luxury Section Attendance': 'int64'})

    if level == 'lower':
        n = 20000
    if level == 'middle':
        n = 20000
    if level == 'upper':
        n = 25000
    if level == 'luxury':
        n = 5000

    data['pct'] = data[' Luxury Section Attendance'] / n
    data['target'] = np.where(data['pct'] == 1, 15, np.where(data['pct'] == 0, -6.98, np.log(data['pct'] / (1 - data['pct']))))

    model_df = data[[' Home Win %', ' Visitor Win %', ' Home Market Index', ' Home Stars Index', ' Visitor Stars Index', ' Home AFP', ' Visitor AFP', ' Ticket Price', 'target']]

    return(model_df)

model_df = end_value(data=df, level='middle')
model_data = model_df[[' Home Win %', ' Visitor Win %', ' Home Market Index', ' Home Stars Index', ' Visitor Stars Index', ' Home AFP', ' Visitor AFP', ' Ticket Price']]
model_y = model_df[['target']]

reg = lr().fit(model_data, model_y)

import pickle
pickle.dump(reg, open('linear_models/luxury.sav', 'wb'))