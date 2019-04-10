# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 18:52:20 2019

@author: JB - definition of the operation associated to a single neuron with 2 inputs. The sigmoid is the
activated function.
"""
import numpy as np    

def single_neuron(X,W,b):
    Y = W[0]*X[0] + W[1]*X[1];
    Y = sigmoid(Y)+b
    return Y

# Definition of the sigmoid function for the training
# ---------------------------------------------------

def sigmoid(X):
    Y = 1 / (1 + np.exp(-X))
    return Y