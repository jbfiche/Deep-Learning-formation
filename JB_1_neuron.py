# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:25:18 2019

@author: JB - This program is an example of function to train one layer of neurons and demonstrate its 
limitation. The idea is to use this very simple network to separate two populations of data defined by their
positions (x,y). Since there are two populations, we are defining two outputs. The network is therefore
composed of two neurons.
"""

# Definition of the sigmoid function that is used here as the activation function.

import random
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from Training_set_circular import Training_set_circular
from Training_set_linear import Training_set_linear
from Training_SGD_no_batch import DeltaSGD
from Training_SGD_batch import DeltaSGD_batch
from Single_neuron import single_neuron

# Choose the type of training/testing data
# ----------------------------------------

Method = input('Choose between a linear training set (L) or a circular one (C) :')

if Method == 'C':

# Definition of a training data where the two classes are simply defined by either d>radius or d<radius
# -----------------------------------------------------------------------------------------------------

    D = 7;
    Training_set, Testing_set, X1, X2, Y1, Y2 = Training_set_circular(D,250)   

# Definition of a training data where the two classes are simply separated by a straight line
# --------------------------------------------------------------------------------------------

else:
    a = 0.1;
    b = 1;
    Training_set, Testing_set, X1, X2, Y1, Y2 = Training_set_linear(a,b,250) 
    
Ex = np.mean(Training_set[:,0])
Ey = np.mean(Training_set[:,1])
stdx = np.std(Training_set[:,0])
stdy = np.std(Training_set[:,1])

# Normalization of the training set
# ---------------------------------

Training_set[:,0] = (Training_set[:,0]-Ex)/stdx
Training_set[:,1] = (Training_set[:,1]-Ey)/stdy
Testing_set[:,0] = (Testing_set[:,0]-Ex)/stdx
Testing_set[:,1] = (Testing_set[:,1]-Ey)/stdy

Xmin = math.floor(min(Training_set[:,0]))
Xmax = math.ceil(max(Training_set[:,0]))

# Plot the training set
# ---------------------

plt.plot(X1, Y1, 'r.')
plt.plot(X2, Y2, 'b.')
plt.show()    

# Train the network using the SGD method:
# In "Training_data", each row represents a set of inputs (the two first values) and the expected output as the
# last element. The number of loops is the number of epochs used for the training. During the training, all the 
# values of the weigths are saved in the list "W_all".
# ---------------------------------------------------

# Define randomly the first value of the weights

W = np.array([0,0,0,0])
W[0] = random.uniform(-1, 1)
W[1] = random.uniform(-1, 1)
W[2] = random.uniform(-1, 1)
W[3] = random.uniform(-1, 1)

biais = np.array([0,0])
biais[0] = random.uniform(-1, 1)
biais[1] = random.uniform(-1, 1)

W_batch = W
biais_batch = biais

W0_all = []
W1_all = []
E_all = []
W0_batch_all = []
W1_batch_all = []
E_batch_all = []

# Define the number of iterations to train the neuron as well as the learning
# rate
#-----

Nepoc = 2000
alpha = 0.01
alpha_batch = 0.4

# The following for loop is used to train the network with the SGD method.

for x in range(0, Nepoc):
    W,e,biais = DeltaSGD(Training_set,W,biais,alpha)
    W0_all.append(W[0])
    W1_all.append(W[1])
    E_all.append(e)
        
# The following loop is training the same network using the batch method.
        
for x in range(0, Nepoc):
    W_batch,e,biais_batch = DeltaSGD_batch(Training_set,W_batch,biais_batch,alpha_batch)
    W0_batch_all.append(W_batch[0])
    W1_batch_all.append(W_batch[1])
    E_batch_all.append(e)
    
   
# Plot the error at each training step for the two methods.
# --------------------------------------------------------

plt.plot(E_all, "r-")
plt.plot(E_batch_all, "b--")
X_zero = [0, Nepoc]
Y_zero = [0, 0]
plt.plot(X_zero, Y_zero, "k:")
plt.show()

# Analyze the trained network using the testing data
# --------------------------------------------------

X_all_1 = []
Y_all_1 = []
X_all_2 = []
Y_all_2 = []
X_all_batch_1 = []
Y_all_batch_1 = []
X_all_batch_2 = []
Y_all_batch_2 = []

for data_row in Testing_set:
    
    x = data_row[0:2]
    W1 = np.array([W[0],W[1]])
    W2 = np.array([W[2],W[3]])
    y1 = single_neuron(x,W1,0)
    y2 = single_neuron(x,W2,0)
    
    W1_batch = np.array([W_batch[0],W_batch[1]])
    W2_batch = np.array([W_batch[2],W_batch[3]])
    y1_batch = single_neuron(x,W1_batch,0)
    y2_batch = single_neuron(x,W2_batch,0)
    
    if y1 < y2:
        X_all_1.append(data_row[0])
        Y_all_1.append(data_row[1])
    else:
        X_all_2.append(data_row[0])
        Y_all_2.append(data_row[1])
        
    if y1_batch < y2_batch:
        X_all_batch_1.append(data_row[0])
        Y_all_batch_1.append(data_row[1])
    else:
        X_all_batch_2.append(data_row[0])
        Y_all_batch_2.append(data_row[1])
        
if Method == 'L':
    X = np.array([-10,10])
    Y = a*X + b
else:
    Angle = np.linspace( 0, 2*pi, 100 )
    X = np.cos(Angle)*D
    Y = np.sin(Angle)*D
    
X = (X-Ex)/stdx
Y = (Y-Ey)/stdy  
    
plt.plot(X_all_1, Y_all_1, 'r.')
plt.plot(X_all_2, Y_all_2, 'b.')
plt.plot(X,Y,'--k')
plt.show()
plt.plot(X_all_batch_1, Y_all_batch_1, 'r+')
plt.plot(X_all_batch_2, Y_all_batch_2, 'b+')
plt.plot(X,Y,'--k')
plt.show() 