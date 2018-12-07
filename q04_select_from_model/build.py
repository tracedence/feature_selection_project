# %load q04_select_from_model/build.py
# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

np.random.seed(9)
# Your solution code here
def select_from_model(df):
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    Random = RandomForestClassifier()
    best_features = SelectFromModel(Random)
    best_features.fit(x,y)
    feature_name = list(x.columns[best_features.get_support()])
    return feature_name
select_from_model(data)


