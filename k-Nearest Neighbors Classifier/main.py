#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 15:02:02 2020

@author: safak
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets

class KNN(object):

    """
    This algorithm based on this motto: 

    'TELL ME ABOUT YOUR FRIEND 
    AND I WILL TELL YOU WHO YOU ARE'

    """
    
    def __init__(self,x,y,K):
        
        self.x   =    x
        self.y   =    y 
        self.K   =    K
        
    def train_test_split(self):
        
        '''
        The Training Data is 80% of
        pure data,
        The Test Data is 20% of
        pure data
        '''
        train_pct_index           =  int(0.8 * len(self.x))        
        self.X_train, self.X_test = self.x[:train_pct_index], self.x[train_pct_index:] 
        self.Y_train, self.Y_test = self.y[:train_pct_index], self.y[train_pct_index:]
    
    def statistics(self):
        
        mean_x , mean_y = np.mean(self.x) , np.mean(self.y)
        var_x , var_y   = np.var(self.x) , np.var(self.y)
        std_x , std_y   = np.sqrt(var_x) , np.sqrt(var_y)
        return(mean_x,mean_y,var_x,var_y,std_x,std_y)
        
    def model_evaluation(self,actual_data,predicted_data):
        
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
    
    def euclidean_distance(self,unseen_data):
        
        distance_vector = np.zeros([self.X_train.shape[0],unseen_data.shape[0]*2])
        k=0
        for i in range(0,unseen_data.shape[0]*2,2):
            for j in range(self.X_train.shape[0]):
                
                distance_vector[j,i] = np.linalg.norm(unseen_data[k,:]-self.X_train[j,:])
                distance_vector[j,i+1] = self.Y_train[j]
            k=k+1
        return distance_vector
    
    def model_testing(self):
        
        self.train_test_split()
        mean_x, mean_y, var_x, var_y, std_x, std_y = self.statistics()
        metrics = {}
        distance_vector = self.euclidean_distance(self.X_test)
        sorted_distance=np.zeros([int(self.K),int(self.X_test.shape[0])*2])
        probability=np.zeros(self.X_test.shape[0])
        k=0
        for i in range(0,self.X_test.shape[0]*2,2):
            tmp=distance_vector[:,i:i+2]
            tmp=tmp[tmp[:,0].argsort()]
            sorted_distance[:,i] = tmp[0:self.K,0]
            sorted_distance[:,i+1] = tmp[0:self.K,1]
            for j in range(self.K): 
                probability[k] = (probability[k] + sorted_distance[j,i+1])
            probability[k] = probability[k] / self.K
            k=k+1
        metrics = self.model_evaluation(self.Y_test,probability)       
        return sorted_distance, distance_vector, probability, metrics
    
    def predict(self, unseen_data):
        
        distance_vector = self.euclidean_distance(unseen_data)
        sorted_distance=np.zeros([int(self.K),int(unseen_data.shape[0])*2])
        probability=np.zeros(unseen_data.shape[0])
        k=0
        for i in range(0,unseen_data.shape[0]*2,2):
            tmp=distance_vector[:,i:i+2]
            tmp=tmp[tmp[:,0].argsort()]
            sorted_distance[:,i] = tmp[0:self.K,0]
            sorted_distance[:,i+1] = tmp[0:self.K,1]
            for j in range(self.K): 
                probability[k] = (probability[k] + sorted_distance[j,i+1])
            probability[k] = probability[k] / self.K
            k=k+1
        return sorted_distance, distance_vector, probability
 
if __name__ == "__main__":

    x,y = sklearn.datasets.make_moons(200,noise=0.50)

    plt.figure(figsize=(5,5))   
    plt.xlim(-2,3)
    plt.ylim(-1.5,2)
    plt.scatter(x[:,0],x[:,1], c=y)
    plt.show()
    metrics = {}
    model=KNN(x,y,3)
    sorted_distance, distance, test_predicted, metrics=model.model_testing()
