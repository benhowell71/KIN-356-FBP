import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import warnings

def lower_prices(home_win, vist_win, hmi, hsi, vsi, hfp, vfp):

    model = pickle.load(open('linear_models\lower.sav', 'rb'))

    price = []
    attendance = []
    revenue = []

    for i in np.arange(1, 500):

        p = model.predict(np.array([[home_win, vist_win, hmi, hsi, vsi, hfp, vfp, i]]))
        att = np.rint((1 / (np.exp(-p) + 1)) * 20000)
        pft = att * i

        price.append(i)
        attendance.append(att.item())
        revenue.append(pft.item())

    result = pd.DataFrame({'price': price, 'attendace': attendance, 'revenue': revenue})
    print("Here are your top-5 prices for the lower level: ")
    print(result.sort_values(by = 'revenue', ascending=False).head(5))

def middle_prices(home_win, vist_win, hmi, hsi, vsi, hfp, vfp):

    model = pickle.load(open('linear_models\middle.sav', 'rb'))

    price = []
    attendance = []
    revenue = []

    for i in np.arange(1, 500):

        p = model.predict(np.array([[home_win, vist_win, hmi, hsi, vsi, hfp, vfp, i]]))
        att = np.rint((1 / (np.exp(-p) + 1)) * 20000)
        pft = att * i

        price.append(i)
        attendance.append(att.item())
        revenue.append(pft.item())

    result = pd.DataFrame({'price': price, 'attendace': attendance, 'revenue': revenue})
    print("Here are your top-5 prices for the middle level: ")
    print(result.sort_values(by = 'revenue', ascending=False).head(5))

def upper_prices(home_win, vist_win, hmi, hsi, vsi, hfp, vfp):

    model = pickle.load(open('linear_models/upper.sav', 'rb'))

    price = []
    attendance = []
    revenue = []

    for i in np.arange(1, 500):

        p = model.predict(np.array([[home_win, vist_win, hmi, hsi, vsi, hfp, vfp, i]]))
        att = np.rint((1 / (np.exp(-p) + 1)) * 20000)
        pft = att * i

        price.append(i)
        attendance.append(att.item())
        revenue.append(pft.item())

    result = pd.DataFrame({'price': price, 'attendace': attendance, 'revenue': revenue})
    print("Here are your top-5 prices for the upper level: ")
    print(result.sort_values(by = 'revenue', ascending=False).head(5))

def luxury_prices(home_win, vist_win, hmi, hsi, vsi, hfp, vfp):

    model = pickle.load(open('linear_models\luxury.sav', 'rb'))

    price = []
    attendance = []
    revenue = []

    for i in np.arange(1, 500):

        p = model.predict(np.array([[home_win, vist_win, hmi, hsi, vsi, hfp, vfp, i]]))
        att = np.rint((1 / (np.exp(-p) + 1)) * 20000)
        pft = att * i

        price.append(i)
        attendance.append(att.item())
        revenue.append(pft.item())

    result = pd.DataFrame({'price': price, 'attendace': attendance, 'revenue': revenue})
    print("Here are your top-5 prices for the luxury level: ")
    print(result.sort_values(by = 'revenue', ascending=False).head(5))

home_win1 = input("Home Win%: ")
vist_win1 = input("Visitor Win%: ")
hmi1 = input("Home Market Index: ")
hsi1 = input("Home Stars Index: ")
vsi1 = input("Visitor Stars Index: ")
hfp1 = input("Home AFP: ")
vfp1 = input("Visitor AFP: ")

warnings.filterwarnings("ignore")
lower_prices(home_win=home_win1, vist_win=vist_win1, hmi=hmi1, hsi=hsi1, vsi=vsi1, hfp=hfp1, vfp=vfp1)
warnings.filterwarnings("ignore")
middle_prices(home_win=home_win1, vist_win=vist_win1, hmi=hmi1, hsi=hsi1, vsi=vsi1, hfp=hfp1, vfp=vfp1)
warnings.filterwarnings("ignore")
upper_prices(home_win=home_win1, vist_win=vist_win1, hmi=hmi1, hsi=hsi1, vsi=vsi1, hfp=hfp1, vfp=vfp1)
warnings.filterwarnings("ignore")
luxury_prices(home_win=home_win1, vist_win=vist_win1, hmi=hmi1, hsi=hsi1, vsi=vsi1, hfp=hfp1, vfp=vfp1)