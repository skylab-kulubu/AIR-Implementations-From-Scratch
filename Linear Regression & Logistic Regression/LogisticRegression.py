#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")       
df = df.sample(frac=1).reset_index(drop=True)

df_train = df.iloc[ :int(df.shape[0]*0.7), : ]
df_test  = df.iloc[ :int(df.shape[0]*0.3), : ]

x_train = df_train.iloc[:, :-1]
x_train = (x_train - np.mean(x_train)) / np.std(x_train)
x_train['ones'] = 1
x_train = np.array(x_train)

y_train = np.array(df_train.iloc[:,-1])
y_train.shape = (y_train.shape[0],1)

x_test = df_test.iloc[:, :-1]
x_test = (x_test - np.mean(x_test)) / np.std(x_test)
x_test['ones'] = 1
x_test = np.array(x_test)

y_test = np.array(df_test.iloc[:,-1])
y_test.shape = (y_test.shape[0],1)

lr = 0.2
iteration = 1000

thetas = np.zeros(shape=(1,x_train.shape[1]))

x_axis = np.zeros(iteration)

def cost(x, y, thetas):
    cost = (-1/x.shape[0]) * (y * np.log((1/(1 + np.exp(-x @ thetas.T)))) + (1-y)*np.log(1-((1/(1 + np.exp(-x @ thetas.T)))))).sum()
    return cost

loss_train = []
loss_test = []

for i in range(iteration):
    thetas = thetas - (lr) * (1/x_train.shape[0]) * (x_train * ((1/(1 + np.exp(-x_train @ thetas.T))) - y_train)).sum(axis = 0)
    loss_train.append(cost(x_train, y_train, thetas))
    loss_test.append(cost(x_test, y_test, thetas))
    x_axis[i] = i 
    
    if(iteration % 100):
        lr = lr * 0.3

plt.plot(x_axis[0:200],loss_train[0:200],color='red')
plt.plot(x_axis[0:200],loss_test[0:200])

predicts = x_test @ thetas.T
predicts = predicts > 0
predicts_train = x_train @ thetas.T
predicts_train = predicts_train > 0
accuracy_test = (predicts == y_test).mean()
accuracy_train = (predicts_train == y_train).mean()
