# -*- coding: utf-8 -*-
"""
Stochastic gradient descent to factorize sparse matrices while
ignoring 

Created on Tue May 02 22:33:56 2017

@author: Chris Pierse
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt


def nonzero_factorization(data, d=20, batch_size=64, eps = 0.001, svd_start = True, 
                          verbose=True,min_eps=10**-7,eps_drop_frac=0.5):
    # We will be slicing out rows, so make sure data is a csr_matrix.
    assert type(data) == scipy.sparse.csr.csr_matrix
    height,width = data.shape
    if svd_start:
        # Initialize matrix with the truncated svd which includes the zeros:
        # This increases the likelihood of convergence.
        U,s,Vt = scipy.sparse.linalg.svds(data,k=d)
        s = np.diag(s)
        U = np.dot(U,np.sqrt(s))
        Vt = np.dot(np.sqrt(s),Vt)
        # Add noise to let gradient descent work:
        U[U==0] = np.random.randn(len(U[U==0]))
        Vt[Vt==0] = np.random.randn(len(Vt[Vt==0]))
    else:
        # Initialize matrix with noise:
        U = np.random.randn(height,d)
        Vt = np.random.randn(d,width)
    # Variables to track costs:
    costs = [100]*100
    avg_costs = [np.Inf]*20
    recorded_costs = []
    for i in range(10**6):
        # Choose random rows and slice the data:
        rows = np.random.choice(height,batch_size)
        data_slice = data[rows,:].todense()
        # Determine which indices which should be used:
        valid_entries = data_slice!=0
        U_slice = np.copy(U[rows,:])
        # Calculate error and update:
        error = np.multiply(np.dot(U_slice,Vt)-data_slice,valid_entries)
        U[rows,:] -= 1.0/batch_size*eps*np.dot(error,np.transpose(Vt))
        Vt -= 1.0/batch_size*eps*np.dot(np.transpose(U_slice),error)
        costs[i%100] = np.sum(np.power(error,2))
        # Track costs and drop learning rate if cost is just bouncing around.
        if i%100==99:
            avg_costs[np.int(np.round(np.float(i%2000)/100))-1] = np.mean(costs)
            print(np.mean(costs))
            if i%2000==1999:
                print('Average cost: ' + str(np.mean(avg_costs)))
                recorded_costs.append(np.mean(avg_costs))
                if len(recorded_costs)>3:
                    plt.plot(range(len(recorded_costs)),recorded_costs)
                    plt.show()
                    if np.mean(recorded_costs[-3:-1])-recorded_costs[-1]<0:
                        new_eps =  max([eps*eps_drop_frac, min_eps])
                        if new_eps==eps:
                            print('Learning rate at minimum, ending program')
                            break
                        else:
                            print('Dropping the learning rate')
                            eps = new_eps
    return U,Vt
