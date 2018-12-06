# %load q02_best_k_features/build.py
# Default imports

import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression


# Write your solution here:

def percentile_k_features(df, K=20):
    x = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    #selecting features on the basis of p-value i.e whose value less than percentile is true
    best_feature = SelectPercentile(f_regression, percentile=K)
    #selecting best features from X
    best_feature.fit_transform(x,y)
    #creating dataframe from score, get_support, result
    d =  {'support': best_feature.get_support(),'values':best_feature.scores_}
    df1 = pd.DataFrame(d,index = x.columns)
    #sorting values according get_support
    df1 = df1.sort_values('values', ascending=False)
    #selecting only rows whose value of support is True
    col = df1[df1.support].index
    return list(col) # returning list of features 
percentile_k_features(data ,20)


