# %load q03_rf_rfe/build.py
# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code hered
def rf_rfe(df):
    
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    #creating a model
    Ra = RandomForestClassifier()
    rf = RFE(Ra)
    #
    rf.fit(x,y) 
    most_sig = list(x.columns[rf.support_])
    return most_sig
rf_rfe(data)


