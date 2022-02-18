import pandas as pd
import numpy as np

# 70000 seats total
# Upper: 25000
# Middle: 20000
# Lower: 20000
# Luxury: 5000

seats = ['lowerlevel.csv', 'luxurylevel.csv', 'middlelevel.csv', 'upperlevel.csv']

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