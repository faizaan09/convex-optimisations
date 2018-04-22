######
######  This file includes different functions used in HW4
######

import numpy as np
import matplotlib.pyplot as plt
import time
import random
import math

import pdb; pdb.set_trace()

def svm_objective_function(w, features, labels, order):
    n=len(labels)
    if order==0:
        # value = ( TODO: value )
        value = 1/n * np.sum(np.maximum(1-np.multiply((features*w),labels),np.zeros_like(n)))
        return value
    elif order==1:
        # value = ( TODO: value )
        value = np.inf

        # subgradient = ( TODO: subgradient )
        hinge_loss = 1 - np.multiply(labels,features*w)
        hinge_loss = np.maximum(np.zeros_like(labels), hinge_loss)
        hinge_loss = np.asarray(hinge_loss).squeeze()
        all_subgradient_evals = - np.asarray(np.multiply(labels,features))
        
        subgradient = 1/n *np.sum(all_subgradient_evals[hinge_loss!=0,:],axis=0)
        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
    
def svm_objective_function_stochastic(w, features, labels, order, minibatch_size):
    n=len(labels)
    batch = np.random.randint(n,size=minibatch_size)
    features = features[batch]
    labels = labels[batch]
    if order==0:
        # value = ( TODO: value )
        value = 1/n * np.sum(np.maximum(1-np.multiply((features*w),labels),0))
        return value
    elif order==1:
        # value = ( TODO: value )
        value = np.inf
        # subgradient = ( TODO: subgradient )
        hinge_loss = 1 - np.multiply(labels,features*w)
        hinge_loss = np.maximum(np.zeros_like(labels), hinge_loss)
        hinge_loss = np.asarray(hinge_loss).squeeze()
        all_subgradient_evals = - np.asarray(np.multiply(labels,features))
        
        subgradient = 1/minibatch_size* np.sum(all_subgradient_evals[hinge_loss!=0,:],axis=0)

        return (value, subgradient)
    else:
        raise ValueError("The argument \"order\" should be 0 or 1")
