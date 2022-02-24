import pandas as pd
import numpy as np

# 70000 seats total
# Upper: 25000
# Middle: 20000
# Lower: 20000
# Luxury: 5000

seats = ['lowerlevel.csv', 'luxurylevel.csv', 'middlelevel.csv', 'upperlevel.csv']

df = pd.read_csv('data/lowerlevel.csv')

df.rename(columns={c: c.strip() for c in df.columns.values.tolist()}, inplace=True) 

def end_value(data, level):
    data = data.dropna()

    if level == 'lower':
        n = 20000
    if level == 'middle':
        n = 20000
    if level == 'upper':
        n = 25000
    if level == 'luxury':
        n = 5000

    data['pct'] = data['Attendance'] / n
    data['target'] = np.where(data['pct'] == 1, 15, np.log(data['pct'] / (1 - data['pct'])))

    model_df = data[['Home Win %', 'Visitor Win %', 'Home Market Index', 'Home Stars Index', 'Visitor Stars Index', 'Home AFP', 'Visitor AFP', 'Ticket Price', 'target']]

    return(model_df)

end_value(data=df, level='lower')


for z in seats:

    if z.isin('lowerlevel.csv', 'middlelevel.csv'):
        n = 20000
    elif z == 'upperlevel.csv':
        n = 25000
    elif z == 'luxurylevel.csv':
        n = 5000

    file = 'data/' + z
    df = pd.read_csv(file)

    print(df)

    print(z)