# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:43:14 2019

@author: JB
"""

from Single_neuron import single_neuron
import numpy as np

def DeltaSGD_batch(Training_data,W,b,alpha):

    dW0 = 0
    dW1 = 0
    dW2 = 0
    dW3 = 0
    db0 = 0
    db1 = 0
    
    E = 0
    L = len(Training_data)
    b = [0,0]
    
    for data_row in Training_data:
        
        x = data_row[0:2]
        
        # training for the first neuron
        # -----------------------------
        
        y_truth = data_row[2]
        W0 = np.array([W[0],W[1]])
        y = single_neuron(x,W0,b[0])
        e0 = y_truth - y
        delta = y*(1-y)*e0
        
        dW0 = dW0 + alpha*delta*x[0]
        dW1 = dW1 + alpha*delta*x[1]
        db0 = db0 + alpha*e0
        
        # training for the second neuron
        # ------------------------------
        
        y_truth = data_row[3]
        W1= np.array([W[2],W[3]])
        y = single_neuron(x,W1,b[1])
        e1 = y_truth - y
        delta = y*(1-y)*e1
        
        dW2 = dW2 + alpha*delta*x[0]
        dW3 = dW3 + alpha*delta*x[1]
        db1 = db1 + alpha*e1
        
        E = E + abs(e0) + abs(e1)
        
    W = [W[0]+dW0/L, W[1]+dW1/L, W[2]+dW2/L, W[3]+dW3/L]
    b = [b[0]+db0/L, b[1]+db1/L]
    E = E/L
    
    return W,E,b
