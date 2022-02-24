import numpy as np

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