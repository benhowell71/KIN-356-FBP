import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

base = pd.read_csv('base_data.csv')
prices = np.arange(1, 501, step=1)
prices = pd.DataFrame(prices, columns=[' Ticket Price'])
prices['key'] = 0
df = base.merge(prices, on = 'key', how='outer')
df = df.drop(columns=['key'])

def lower_prices(data):
    lower_model = joblib.load('model_objects/lower_model.pkl')

    results = lower_model.predict(df)
    results = 1 / (np.exp(-results) + 1)
    results = pd.DataFrame(results, columns=['pred_pct']).join(prices)
    results['profit'] = (results['pred_pct'] * 20000) * results['Ticket Price']

    return results

def middle_prices(data):
    middle_model = joblib.load('model_objects/middle_model.pkl')
    results = middle_model.predict(df)
    results = 1 / (np.exp(-results) + 1)
    results = pd.DataFrame(results, columns=['pred_pct']).join(prices)
    results['profit'] = (results['pred_pct'] * 20000) * results['Ticket Price']

    return results

def upper_prices(data):
    upper_model = joblib.load('model_objects/upper_model.pkl')
    results = upper_model.predict(df)
    results = 1 / (np.exp(-results) + 1)
    results = pd.DataFrame(results, columns=['pred_pct']).join(prices)
    results['profit'] = (results['pred_pct'] * 25000) * results['Ticket Price']

    return results

def luxury_prices(data):
    luxury_model = joblib.load('model_objects/luxury_model.pkl')
    results = luxury_model.predict(df)
    results = 1 / (np.exp(-results) + 1)
    results = pd.DataFrame(results, columns=['pred_pct']).join(prices)
    results['profit'] = (results['pred_pct'] * 5000) * results['Ticket Price']

    return results

lower_prices(data=df)
pred = middle_prices(data=df)
upper_prices(data=df)
luxury_prices(data=df)

pred['pred_pct'].hist()
plt.show()