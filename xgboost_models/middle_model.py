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

df = pd.read_csv('data/MiddleAttendance.csv')

# df.rename(columns={c: c.strip() for c in df.columns.values.tolist()}, inplace=True) 
def end_value(data, level):
    data = data.reset_index()
    data.columns = data.iloc[0]
    data = data.drop(data.index[0])
    data = data.dropna()

    data = data.astype({' Home Win %': 'float64', ' Visitor Win %': 'float64', ' Home Market Index': 'float64', ' Home Stars Index': 'int64', ' Visitor Stars Index': 'int64', ' Home AFP': 'int64', ' Visitor AFP': 'int64', ' Ticket Price': 'float64', ' Middle Section Attendance': 'int64'})

    if level == 'lower':
        n = 20000
    if level == 'middle':
        n = 20000
    if level == 'upper':
        n = 25000
    if level == 'luxury':
        n = 5000

    data['pct'] = data[' Middle Section Attendance'] / n
    data['target'] = np.where(data['pct'] == 1, 15, np.where(data['pct'] == 0, -6.98, np.log(data['pct'] / (1 - data['pct']))))

    model_df = data[[' Home Win %', ' Visitor Win %', ' Home Market Index', ' Home Stars Index', ' Visitor Stars Index', ' Home AFP', ' Visitor AFP', ' Ticket Price', 'target']]

    return(model_df)

model_df = end_value(data=df, level='middle')
model_data = model_df[[' Home Win %', ' Visitor Win %', ' Home Market Index', ' Home Stars Index', ' Visitor Stars Index', ' Home AFP', ' Visitor AFP', ' Ticket Price']]
model_y = model_df[['target']]
x_train, x_test, y_train, y_test = train_test_split(model_data, model_y, test_size = 0.01, random_state=123)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

params = {'max_depth': [4, 6, 8, 10],
           'learning_rate': [0.05, 0.2, 0.5],
           'n_estimators': [250]}

xgbr = xgb.XGBRegressor(seed = 20)
clf = GridSearchCV(estimator=xgbr,
                   param_grid=params,
                   scoring='neg_root_mean_squared_error', 
                   verbose=3,
                   cv=10)

clf.fit(model_data, model_y)

model = clf.best_estimator_

joblib.dump(model, 'model_objects/middle_model.pkl')

results = model.predict(x_test)
results = 1 / (np.exp(-results) + 1)
results = pd.DataFrame(results, columns=['pred_pct'])
y_test_val = 1 / (np.exp(-y_test) + 1)
y_test_val = y_test_val.reset_index()
results = results.join(y_test_val)

mean_squared_error(results['target'], results['pred_pct'], squared=False)



reg = lr().fit(model_data, model_y)



reg.predict(np.array([[1, 0, 1.6, 4, 1, 203, 193, 10]]))