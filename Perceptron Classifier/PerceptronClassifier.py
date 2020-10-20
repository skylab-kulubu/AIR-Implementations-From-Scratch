#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:28:38 2020

@author: safak
"""
#%%
import numpy as np
from numpy import log
import matplotlib.pyplot as plt

class Perceptron(object):
    
    def __init__(self,X,Y,learning_rate,epochs):
        
        self.x       = X
        self.y       = Y
        self.lr      = learning_rate
        self.epochs  = epochs
        self.bias    = 0.7
        self.weights = np.random.randn(1,self.x.shape[1]) * 0.8
    
    def __init__matrix(self):
        
        bias_ones       =  np.ones(self.x.shape[0])
        bias_ones.shape =  (self.x.shape[0],1)
        self.x          =  np.concatenate((bias_ones,self.x),axis=1)
        self.weights    =  np.insert(self.weights, 0, self.bias)
        
    def Splitting_Data(self):
        '''
        The Training Data is 80% of
        pure data,
        The Test Data is 20% of
        pure data
        '''
        train_pct_index           =  int(0.8 * len(self.x))        
        self.X_train, self.X_test = self.x[:train_pct_index], self.x[train_pct_index:] 
        self.Y_train, self.Y_test = self.y[:train_pct_index], self.y[train_pct_index:]
        
    def Statistics(self):

        mean_x , mean_y = np.mean(self.x) , np.mean(self.y)
        var_x , var_y   = np.var(self.x) , np.var(self.y)
        std_x , std_y   = np.sqrt(var_x) , np.sqrt(var_y)
        return(mean_x,mean_y,var_x,var_y,std_x,std_y)
        
    def Sigmoid(self,Z):
        
        return (1/(1+np.exp(-Z)))
    
    def D_Sigmoid(self,Z):
        
        return ((1-self.Sigmoid(Z)) * self.Sigmoid(Z))
    
    def SingleForwardPropagation(self,X):
        
        y_hat = self.Sigmoid(np.dot(X,self.weights.T) + self.bias )
        
        return y_hat
    
    def Cost(self,Y,y_hat):
        
        return (1/(2*Y.shape[0])) * np.sum((Y - y_hat)**2)
    
    def Train(self):
        
#        self.__init__matrix()
        self.Splitting_Data()
        mean_x,mean_y,var_x,var_y,std_x,std_y = self.Statistics()
        y_hat = self.SingleForwardPropagation(self.X_train)
        training_cost_storage   =   np.zeros(int(self.epochs))
        test_cost_storage       =   np.zeros(int(self.epochs))
        model= {}
        
        for i in range(self.epochs):
            self.weights = self.weights - ((1/len(self.X_train))*self.lr) * np.sum(self.X_train *
                          self.D_Sigmoid(y_hat - self.Y_train) * (y_hat - self.Y_train),axis=0)
            self.bias =self.bias - self.lr * (1/(2*self.Y_train.shape[0])) * np.sum((self.Y_train - y_hat)**2)
            y_hat = self.SingleForwardPropagation(self.X_train)
            y_hat_test = self.SingleForwardPropagation(self.X_test)
            training_cost_storage[i] = self.Cost(self.Y_train,y_hat)
            test_cost_storage[i]     = self.Cost(self.Y_test,y_hat_test)
        model['Training Cost'] = training_cost_storage
        model['Test Cost'] = test_cost_storage
        model['Mean of X'] = mean_x
        model['Mean of Y'] = mean_y
        model['Variance of X'] = var_x
        model['Variance of Y'] = var_y
        model['Standard Deviation of X'] = std_x
        model['Standard Deviation of Y'] = std_y
        model['Weight'] = self.weights
        model['Bias'] = self.bias
        
        return model
    
#if __name__ == "__main__":
#    
#    import pandas as pd
#    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
#    df=df.iloc[:100,:]
#    df = df.sample(frac=1).reset_index(drop=True)
#    y = df.iloc[:100, 4].values
#    x = df.iloc[0:100, [0, 2]].values
#    y = np.where(y == 'Iris-setosa', 0, 1)
#    y.shape=(100,1)
#    #%%
#    plt.scatter(x[:,0],x[:,1],c=y[:,0])
#    plt.xlabel('petal length')
#    plt.ylabel('sepal length')
#    plt.legend(loc='upper left')
#    plt.show()
#    #%%
#    
#    obj=Perceptron(x,y,0.05,500)
#    model=obj.Train()    