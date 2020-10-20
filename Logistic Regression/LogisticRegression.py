#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 17:25:19 2020

@author: safak
"""

#%%

import numpy as np
from numpy import log

class LogisticRegression(object):
    
    def __init__(self,x,y,learning_rate,iterations):
        """
        self.epsilon is for divide by zero in log
        that is nan value
        self.epsion is bigger than zero but it is too small.
        """
        self.x             =  np.concatenate((np.ones([x.shape[0],1]),x),axis=1)
        self.y             =  y
        self.iterations    =  iterations
        self.thetas        =  np.zeros([1,self.x.shape[1]])
        self.lr            =  learning_rate
        self.epsilon       =  1e-5 
        
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
        
    def Sigmoid(self,var):
        """
        activation function
        """
        sigmoid= (1/(1+np.exp(-var)))
        return sigmoid
        
    def Cost_Function(self,X,Y):

        cost_funct=(-1/(len(X))) * np.sum(Y.T @ log(self.Sigmoid(X @ self.thetas.T) + self.epsilon) -
                    (1-Y).T @ log(1-self.Sigmoid(X @ self.thetas.T) + self.epsilon ))
        
        return cost_funct
    
    def Predict(self,x):
        
        predicted_data = np.dot(x,self.thetas.T)
        
        return predicted_data
    
    def Model_Evaluation(self,actual_data,predicted_data):
        
        confusion_matrix = np.zeros((2,2))
        metrics= {}
        predicted_data=np.round(predicted_data)
        for i in range(len(actual_data)):
        
            if(actual_data[i]==1 and predicted_data[i]==1):
                confusion_matrix[0][0] += 1
        
            elif(actual_data[i]==1 and predicted_data[i]==0):
                confusion_matrix[0][1] += 1
            
            elif(actual_data[i]==0 and predicted_data[i]==1):
                confusion_matrix[1][0] += 1
        
            elif(actual_data[i]==0 and predicted_data[i]==0):
                confusion_matrix[1][1] += 1
        
        TP                             = confusion_matrix[0][0]
        FN                             = confusion_matrix[0][1]
        FP                             = confusion_matrix[1][0]
        TN                             = confusion_matrix[1][1]
    
        accuracy                       = (TP+TN)/(TP+FN+TN+FP)
        precision                      = TP/(TP+FP)
        recall                         = TP/(TP+FN)
        F_measure                      = 2*TP/(2*TP+FN+FP)
        specifity                      = TN/(TN+FP)
        false_positive_rate            = 1-specifity
        
        metrics['Accuracy']            = accuracy
        metrics['Precision']           = precision
        metrics['Recall']              = recall
        metrics['F Measure']           = F_measure
        metrics['Specifity']           = specifity
        metrics['False Positive Rate'] = false_positive_rate
        metrics['Confusion Matrix']    = confusion_matrix
        
        return metrics
        
    
    def Train(self):
        
        self.Splitting_Data()
        mean_x,mean_y,var_x,var_y,std_x,std_y= self.Statistics()
        model= {}
        metrics_train = {}
        metrics_test  = {}
        training_cost_storage   =   np.zeros(int(self.iterations))
        test_cost_storage       =   np.zeros(int(self.iterations))
        
        for i in range(self.iterations):
            self.thetas = self.thetas - ((1/len(self.X_train)) * self.lr * np.sum(self.X_train * 
                                         (self.Sigmoid(self.X_train @ self.thetas.T + self.epsilon) - self.Y_train),axis=0))
            training_cost_storage[i] = self.Cost_Function(self.X_train,self.Y_train)
            test_cost_storage[i]     = self.Cost_Function(self.X_test,self.Y_test)
        predicted_train = self.Predict(self.X_train)
        predicted_test = self.Predict(self.X_test)
        metrics_train = self.Model_Evaluation(self.Y_train,predicted_train)
#        metrics_test = self.Model_Evaluation(self.Y_test,predicted_test)
        model['Training Cost'] = training_cost_storage
        model['Test Cost'] = test_cost_storage
        model['Thetas'] = self.thetas
        model['Mean of X'] = mean_x
        model['Mean of Y'] = mean_y
        model['Variance of X'] = var_x
        model['Variance of Y'] = var_y
        model['Standard Deviation of X'] = std_x
        model['Standard Deviation of Y'] = std_y
        return(model,metrics_test,metrics_train)
    