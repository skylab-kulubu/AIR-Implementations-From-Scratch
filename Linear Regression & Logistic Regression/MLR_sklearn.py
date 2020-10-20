#Enes Sadi Uysal
#%%
import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('insurance.csv')

df.drop('region', inplace = True, axis = 1)

def sex(each):
    if each == 'male':
        return 0
    else:
        return 1

df['sex'] = df['sex'].apply(sex)

def smoker(each):
    if each == 'yes':
        return 1
    else:
        return 0

df['smoker'] = df['smoker'].apply(smoker)  

df = (df - np.mean(df)) / np.std(df)

x = df.iloc[:, :-1]

y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3)

lm = LinearRegression()

lm.fit(x_train, y_train)

thetas = pd.DataFrame(lm.coef_, x.columns,columns = ["Coeff"])

predictions = lm.predict(x_test)
plt.scatter(y_test,predictions)

loss = metrics.mean_squared_error(y_test, predictions)
variance = metrics.explained_variance_score(y_test, predictions)
r2 = metrics.r2_score(y_test, predictions)