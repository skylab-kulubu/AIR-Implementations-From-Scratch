#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 16:43:01 2020

@author: safak
"""

#%%    
if __name__ == "__main__":
    import sys
    sys.path.append('/home/safak/Desktop/PerceptronClassifier')
    from PerceptronClassifier import Perceptron
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    
    """
    Creating the
    pure data
    """
    
    Epoch=500
    Learning_Rate=0.05
    model = {}
    
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
    df=df.iloc[:100,:]
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.iloc[:100, 4].values
    x = df.iloc[0:100, [0, 2]].values
    y = np.where(y == 'Iris-setosa', 0, 1)
    y.shape=(100,1)
    ones = np.ones(shape=(x.shape[0],1))
    x = np.concatenate((ones,x),axis=1)
     
    #%% 
    """
    Visualization of 
    the data
    """
    plt.scatter(x[:,0],x[:,1],c=y[:,0])
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    plt.show()
    #%% 
    """
    Creating and trainin
    the model
    """
    SinglePerceptronModel=Perceptron(x,y,Learning_Rate,Epoch)
    model=SinglePerceptronModel.Train()
    #%%
    """
    Visualization
    of the test error
    and train error
    """
    x_axis=np.zeros(Epoch)
    for i in range(Epoch):
        x_axis[i]=i
    plt.plot(x_axis,model['Test Cost'],'r',label='Test Cost')
    plt.title("Test Error")
    plt.plot(x_axis,model['Training Cost'],'b',label='Training Cost')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    
