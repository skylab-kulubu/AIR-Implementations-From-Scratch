#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:52:46 2020

@author: safak
"""

#%%

if __name__ == "__main__":
    plt.style.use(['dark_background', 'presentation'])
    import sys
    sys.path.append('/home/safak/Desktop/LinearModel')
    from LinearModel import Linear_Regression
    import numpy as np
    import pandas as pd
    from random import shuffle
    import matplotlib.pyplot as plt
    
    Sample_Size=1338
    Iterations=400
    Learning_Rate=0.01
    tuning_parameter=5
    model = {}
    
    
    data_frame= pd.read_csv('insurance.csv')
    data_frame.drop('region',inplace=True,axis=1)
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    for i in range(len(data_frame)):  # make binary gender
    
        if(data_frame.iloc[i,1]=='female'):                                                                 
        
            data_frame.iloc[i,1]=1
        else:
        
            data_frame.iloc[i,1]=0  

    for i in range(len(data_frame)): # #make binary smokers
    
        if(data_frame.iloc[i,4]=='yes'):
        
            data_frame.iloc[i,4]=1
        else:
        
            data_frame.iloc[i,4]=0

    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    data_frame=(data_frame-np.mean(data_frame))/np.std(data_frame)
    
    x=np.array(data_frame.iloc[:,0:5])
    ones = np.ones(shape=(x.shape[0],1))
    x = np.concatenate((ones,x),axis=1)
    y=data_frame.iloc[:,5].values
    y.shape = (Sample_Size,1)
    
    Linear_Model=Linear_Regression(x,y,Learning_Rate,Iterations,tuning_parameter)
    model=Linear_Model.Training_and_Cost()
    model_Ridge=Linear_Model.Training_with_Ridge()
    x_axis=np.zeros(400)
    
'''
Plotting Test And Training Errors
Including Ridge Regression
'''
    #%%
    for i in range(400):
        x_axis[i]=i
    plt.plot(x_axis,model['Test Cost'],'r',label='Test Cost')
    plt.title("Test Error")
    plt.plot(x_axis,model['Training Cost'],'b',label='Training Cost')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    #plt.title("Training Error")
    #%%
    plt.plot(x_axis,model_Ridge['Test Cost'],'r')
    plt.title('Test Cost (Ridge)')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    #%%
    plt.plot(x_axis,model_Ridge['Training Cost'],'b')
    plt.title('Training Cost')
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()
    
