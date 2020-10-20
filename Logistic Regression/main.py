#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:43:11 2020

@author: safak
"""

#%%

if __name__ == "__main__":
    import sys
    import numpy as np
    sys.path.append('/home/safak/Desktop/LogisticR')
    import matplotlib.pyplot as plt
    import pandas as pd
    from LogisticRegression import LogisticRegression
    
    iterations=100000
    learning_rate=0.005
    
    data_frame= pd.read_csv('train_titanic.csv')
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    
    def filling_age(cols):
        Age = cols[0]
        Pclass = cols[1]
    
        if pd.isnull(Age):

            if Pclass == 1:
                return 37

            elif Pclass == 2:
                return 29

            else:
                return 24

        else:
            return Age
    
    data_frame['Age'] = data_frame[['Age','Pclass']].apply(filling_age,axis=1)#applying function to dataframe.
    data_frame.drop('Cabin',axis=1,inplace=True)#Dropping non-quantitative feature 'Cabin'.
    data_frame.dropna(inplace=True)#dropping all nan-values.

    Sex = pd.get_dummies(data_frame['Sex'],drop_first=True)# Making gender binary.
    Embarked = pd.get_dummies(data_frame['Embarked'],drop_first=True)# Making embarked binary.

    data_frame.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)#Deopping all non-quantitative features.

    data_frame = pd.concat([data_frame,Sex,Embarked],axis=1)# concatenating binary features 'Sex' and 'Embarked'. 
    
    y= np.array(data_frame["Survived"])
    y.shape = (len(y),1)
    x= np.array(data_frame.iloc[:,1:])
       
    ones = np.ones(shape=(x.shape[0],1))
    x = np.concatenate((ones,x),axis=1)
    
    LogisticRegreesionClassifier = LogisticRegression(x,y,learning_rate,iterations)
    model,metrics_test,metrics_train = LogisticRegreesionClassifier.Train()
