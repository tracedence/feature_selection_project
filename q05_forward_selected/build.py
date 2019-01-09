# %load q05_forward_selected/build.py
# Default imports
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
model = LinearRegression()

data = pd.read_csv('data/house_prices_multivariate.csv')

model = LinearRegression()


# Your solution code here



def forward_selected(data, model):
    X  = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    l = []
    
    score = -1000
    c = ''
    variable_1 = []
    variable_2 = []
    column = list(X.columns)
    for i in range(len(column)):
        
        
        for col in column:
            #print(col)
            l.append(col)
            model.fit(X[l],y)
            acc = model.score(X[l],y)
            #print(col, acc)
            if acc > score:
                score = acc
                c = col
            l.pop(len(l)-1)
#         print('  ')
            #print(col,c, score, acc)
        #print('  ')
        if c in l:
            pass
        else:
            l.append(c)
            column.remove(c)
            variable_2.append(c)
        variable_1.append(score)
    return variable_2, variable_1


var1 , var2 = forward_selected(data, model)
var2
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score, accuracy_score
# model = LinearRegression()



# def forward_selected(X,y,i):
#     X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state =i)
#     l = []
    
#     score = -1000
#     c = ''
#     variable_1 = []
#     variable_2 = []
#     column = list(X_train.columns)
#     for i in range(len(column)):
        
        
#         for col in column:
#             #print(col)
#             l.append(col)
#             model.fit(X_train[l],y_train)
#             y_pred = model.predict(X_test[l])
#             acc = r2_score(y_pred, y_test)
#             #print(col, acc)
#             if acc > score:
#                 score = acc
#                 c = col
#             l.pop(len(l)-1)
# #         print('  ')
# #         print(c, score)
#         if c in l:
#             pass
#         else:
#             l.append(c)
#             column.remove(c)
#             variable_2.append(c)
#         variable_1.append(score)
#     return variable_2
# data.columns


   




