#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:52:46 2020

@author: safak
"""
import numpy as np
import matplotlib.pyplot as plt
class Linear_Regression(object): # add dataframe to calculate statistical paramteres (varicance,standard deviation...)
    
    def __init__(self,x,y,learning_rate,iterations,tuning_parameter):
        self.x             =  np.concatenate((np.ones([x.shape[0],1]),x),axis=1)
        self.y             =  y
        self.iterations    =  iterations
        self.thetas        =  np.zeros([1,self.x.shape[1]])
        self.lr            =  learning_rate
        self.tp            =  tuning_parameter
    
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
        
    def Cost_Function(self,X,Y):
        to_be_sum                 =  np.power((X @ self.thetas.T - Y),2)
        total_cost                =  (1/(2*len(X)))*(np.sum(to_be_sum))
        return total_cost
    
    def Ridge_Regression(self,X,Y):
        '''
        RSS is modified by adding the shrinkage quantity.
        Now, the coefficients are estimated by minimizing this function.
        Here, λ (self.tp) is the tuning parameter that decides,
        how much we want to penalize the flexibility of our model. 
        as λ→∞, the impact of the shrinkage penalty grows, 
        and the ridge regression coeﬃcient estimates will approach zero
        '''
        
        to_be_sum_ridge             =  np.power((X @ self.thetas.T - Y),2)
        ridge_sum                   =  np.power(self.thetas,2)
        indices                     =  np.delete(ridge_sum, np.s_[0:1], axis=1)
        total_cost_ridge            =  (1/(2*len(X)))*((np.sum(to_be_sum_ridge)) + self.tp * np.sum(indices))
        return total_cost_ridge 
        
    def Residual_Standard_Error(self):
        '''
        Typically we have a regression model looks like this:
        y = b0 + b1*x1 + ... + bn*xn + E
        where E is an error term independent of X
        if b0 and b1 ... bn are known,
        we still cannot perfectly predict Y using X due to E
        Therefore, we use RSE as a judgement value of the Standard Deviation of E
        '''
        RSS_term                 =  np.power((self.X_train @ self.thetas.T - self.Y_train),2)
        Residual_Standard_Error  =  np.sqrt((1/(len(self.X_train)-2))*np.sum(RSS_term))   
        return(Residual_Standard_Error)
       
    def R2(self):
        '''
        As for the R² metric,
        it measures the proportion of variability in the target
        that can be explained using a feature X.
        Therefore, assuming a linear relationship,
        if feature X can explain (predict) the target,
        then the proportion is high and the R² value will be close to 1.
        If the opposite is true, the R² value is then closer to 0.
        '''
        RSS_term              = np.power((self.X_train  @ self.thetas.T - self.Y_train),2)
        Residual_Squared_Sum  = np.sum(RSS_term)
        TSS_term              = np.power(np.mean(self.Y_train) - self.Y_train,2)
        Total_Sum_of_Squares  = np.sum(TSS_term)    
        R2                    = 1 - (Residual_Squared_Sum/Total_Sum_of_Squares)    
        return R2
    
    def Training_and_Cost(self): 
        self.Splitting_Data()
        model= {}
        mean_x,mean_y,var_x,var_y,std_x,std_y = self.Statistics()
        training_cost_storage   =   np.zeros(int(self.iterations))
        test_cost_storage       =   np.zeros(int(self.iterations))
        for i in range(self.iterations):
            self.thetas         = self.thetas - ((1/len(self.X_train))*self.lr) * np.sum(self.X_train *
                          (self.X_train @ self.thetas.T - self.Y_train),axis=0)
            training_cost_storage[i] = self.Cost_Function(self.X_train,self.Y_train)
            test_cost_storage[i]     = self.Cost_Function(self.X_test,self.Y_test)
            if(i == self.iterations/10):
                self.lr = self.lr * 0.80
        thetas=self.thetas
        Residual_Standard_Error = self.Residual_Standard_Error()
        R2=self.R2()
        model['Training Cost'] = training_cost_storage
        model['Test Cost'] = test_cost_storage
        model['Residual Standard Error'] = Residual_Standard_Error
        model['R2'] = R2
        model['Thetas'] = thetas
        model['Mean of X'] = mean_x
        model['Mean of Y'] = mean_y
        model['Variance of X'] = var_x
        model['Variance of Y'] = var_y
        model['Standard Deviation of X'] = std_x
        model['Standard Deviation of Y'] = std_y
        return(model)
                    
    def Training_with_Ridge(self):
        self.Splitting_Data()
        model={}
        mean_x,mean_y,var_x,var_y,std_x,std_y = self.Statistics()
        training_cost_storage   =   np.zeros(int(self.iterations))
        test_cost_storage       =   np.zeros(int(self.iterations))
        for i in range(int(self.iterations)):
            self.thetas         = self.thetas - ((1/len(self.X_train))*self.lr) * (np.sum(self.X_train *
                          (self.X_train @ self.thetas.T - self.Y_train),axis=0) + self.tp * self.thetas)
            training_cost_storage[i] = self.Ridge_Regression(self.X_train,self.Y_train)
            test_cost_storage[i]     = self.Ridge_Regression(self.X_test,self.Y_test)
            if(i == self.iterations/10):
                self.iterations = self.iterations * 0.80
        Residual_Standard_Error = self.Residual_Standard_Error()
        thetas=self.thetas
        R2=self.R2()
        model['Training Cost'] = training_cost_storage
        model['Test Cost'] = test_cost_storage
        model['Residual Standard Error'] = Residual_Standard_Error
        model['R2'] = R2
        model['Thetas'] = thetas
        model['Mean of X'] = mean_x
        model['Mean of Y'] = mean_y
        model['Variance of X'] = var_x
        model['Variance of Y'] = var_y
        model['Standard Deviation of X'] = std_x
        model['Standard Deviation of Y'] = std_y
        return model
    
