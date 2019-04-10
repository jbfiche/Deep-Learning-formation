# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:07:26 2019

@author: JB - The Single Gradient Descent (SGD) is used to train this neuron. The delta rule is an algorithm used to adjust the 
weights during the training. Here, for each interation, the neuron output "y" is calculated according to the
training data. The value of "y" is compared to the ground truth and a correction of the weights "delta" is
calculated. Therefore, for each iteration, the weights "W" are modified. Note also that "alpha" is an indication
of the training rate. 
"""

# The Single Gradient Descent (SGD) is used to train this neuron. The delta rule is an algorithm used to adjust the 
# weights during the training. Here, for each interation, the neuron output "y" is calculated according to the
# training data. The value of "y" is compared to the ground truth and a correction of the weights "delta" is
# calculated. Therefore, for each iteration, the weights "W" are modified. Note also that "alpha" is an indication
# of the training rate. 

from Single_neuron import single_neuron
import numpy as np

def DeltaSGD(Training_data,W,b,alpha):
    
    for data_row in Training_data:
        
        x = data_row[0:2]
        
        # training for the first neuron
        # -----------------------------
        
        y_truth = data_row[2]
        W0 = np.array([W[0],W[1]])
        y = single_neuron(x,W0,b[0])
        e0 = y_truth - y
        delta = y*(1-y)*e0
        
        dW0 = alpha*delta*x[0]
        dW1 = alpha*delta*x[1]
        db0 = alpha*e0
        
        # training for the second neuron
        # ------------------------------
        
        y_truth = data_row[3]
        W1 = np.array([W[2],W[3]])
        y = single_neuron(x,W1,b[1])
        e1 = y_truth - y
        delta = y*(1-y)*e1
        
        dW2 = alpha*delta*x[0]
        dW3 = alpha*delta*x[1]
        db1 = alpha*e1
        
        W = [W[0]+dW0, W[1]+dW1, W[2]+dW2, W[3]+dW3]
        b = [b[0]+db0, b[1]+db1]
        E = abs(e0)+abs(e1)
  
    return W,E,b
