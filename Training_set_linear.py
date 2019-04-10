# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:34:44 2019

@author: JB - Program dedicated to the creation of a training set based on two classes.
The two classes are simply defined according to the distance from 0. If d<5, any points will
belong to class #1, else it will belong to class #2.
"""

# Definition of a training data where the two classes are simply defined by Y<aX+b or Y>aX+b
# ------------------------------------------------------------------------------------------
   
def Training_set_linear(a,b,N):
 
    import random
    import numpy as np
    
    n = 0
    Training_data = np.zeros([N,4])
    Testing_data = np.zeros([N,2])
    X1 = np.array([])
    X2 = np.array([])
    Y1 = np.array([])
    Y2 = np.array([])
    
    for n in range(0,N):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        Training_data[n,0] = x
        Training_data[n,1] = y
        
        if y < a*x+b :
            Training_data[n,2] = 1
            Training_data[n,3] = 0
            X1 = np.append(X1, x)
            Y1 = np.append(Y1, y)
        else : 
            Training_data[n,2] = 0
            Training_data[n,3] = 1
            X2 = np.append(X2, x)
            Y2 = np.append(Y2, y)
            
    for n in range(0,N):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)       
        Testing_data[n,0] = x
        Testing_data[n,1] = y  
        
    return Training_data, Testing_data, X1, X2, Y1, Y2
  