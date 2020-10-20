import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

originalDF = pd.read_csv("kc_house_data.csv")

x = np.array(originalDF["sqft_living"])
y = np.array(originalDF["price"])

x.shape = (x.shape[0],1)
y.shape = (y.shape[0],1)

x = (x- x.mean())/x.std()
y = (y- y.mean())/y.std()

m = len(x)
learning_rate = 0.2

T0 = 0
T1 = 0

for i in range(1,1000):
    tmp0 = T0 - (1/m)*(learning_rate)*((T0 + T1*x) - y).sum()
    tmp1 = T1 - (1/m)*(learning_rate)*(((T0 + T1*x) - y)*x).sum()
    T0 = tmp0
    T1 = tmp1

plt.scatter(x,y)
plt.plot(x, T0 + T1*x)
     

