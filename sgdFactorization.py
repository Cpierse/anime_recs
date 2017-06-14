# -*- coding: utf-8 -*-
"""
Stochastic gradient descent to factorize sparse matrices. Ignores
elements with zero values.

Created on Tue May 02 22:33:56 2017

@author: Chris Pierse
"""

import numpy as np
import scipy
from matplotlib import pyplot as plt


def nonzero_factorization(data, d=20, batch_size=64, eps = 10**-4, svd_start = True, 
                          verbose=True,min_eps=10**-7,eps_drop_frac=10**-0.5,
                          lamb = 0, bias=False, implicit=False):
    ''' Matric factorization where the zero elements are not considered'''
    # We will be slicing out rows, so make sure data is a csr_matrix.
    assert type(data) == scipy.sparse.csr.csr_matrix
    height,width = data.shape
    U_best, Vt_best,Y_best,err_best = [],[],[],np.inf
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
    # Include a term similar to U that accounts for the fact that
    # the user rated an item at all:
    if implicit:
        Y = np.random.randn(data.shape[1],d)/10
    else:
        Y=[]
        #U -= np.dot(Fbool/(10**-8+np.dot(Frow,np.ones((1,width)))), Y)
    # Include a user and anime bias in the results:
    if bias:
        U = np.append(U,np.random.random((U.shape[0],2)),axis=1)
        Vt = np.append(Vt,np.random.random((2,Vt.shape[1])),axis=0)
        U[:,-1] = 1
        Vt[-2,:] = 1
        if implicit:
            Y = np.append(Y,np.zeros((Y.shape[0],2)),axis=1)
    # Variables to track costs:
    N1,N2 = 100,1000
    costs = [100]*N1
    avg_costs = [np.Inf]*10
    recorded_costs = []
    for i in range(10**6):
        # Choose random rows and slice the data:
        rows = np.random.choice(height,batch_size)
        data_slice = data[rows,:].todense()
        # Determine which indices which should be used:
        valid_entries = data_slice!=0
        U_slice = np.copy(U[rows,:])
        
        # Handle calculation if implicit is True
        if implicit:
            # Calculate the F terms:
            #F_bot = (10**-8+np.dot(np.sqrt(np.sum((data_slice!=0),axis=1)),np.ones((1,width))))
            F = (data_slice!=0)/(10**-8+np.dot(np.sqrt(np.sum((data_slice!=0),axis=1)),np.ones((1,width))))
            # Get previous FY slice and calculate error
            FY_slice = np.dot(F, Y)
            error = np.multiply(np.dot(U_slice+FY_slice,Vt)-data_slice,valid_entries)
            # Update according to that error:
            if lamb:
                U[rows,:] -= eps*(np.dot(error,np.transpose(Vt)) + lamb*U_slice)
                Y-= eps*(np.dot(np.transpose(F), np.dot(error,np.transpose(Vt))) + lamb*Y)
                Vt -= eps*(np.dot(np.transpose(U_slice+FY_slice),error)+lamb*Vt)
                costs[i%N1] = np.sum(np.power(error,2))
            else:
                U[rows,:] -= eps*np.dot(error,np.transpose(Vt))
                Y-= eps*np.dot(np.transpose(F), np.dot(error,np.transpose(Vt))) 
                #Y-= eps*np.dot(np.transpose(1/F_bot), np.dot(error,np.transpose(Vt)))
                Vt -= eps*np.dot(np.transpose(U_slice+FY_slice),error)
                costs[i%N1] = np.sum(np.power(error,2))
        else:
            # Calculate error and update:
            error = np.multiply(np.dot(U_slice,Vt)-data_slice,valid_entries)
            # Handle regularization:
            if lamb:
                U[rows,:] -= eps*(np.dot(error,np.transpose(Vt)) + lamb*U_slice)
                Vt -= eps*(np.dot(np.transpose(U_slice),error)+lamb*Vt)
                costs[i%N1] = np.sum(np.power(error,2)) + \
                               lamb/2*(np.sum(U_slice**2)+np.sum(Vt**2))
            else:
                U[rows,:] -= eps*np.dot(error,np.transpose(Vt))
                Vt -= eps*np.dot(np.transpose(U_slice),error)
                costs[i%N1] = np.sum(np.power(error,2))
        # Reset the bias columns:
        if bias:
            U[:,-1] = 1
            Vt[-2,:] = 1
            if implicit:
                Y[:,-2:] = 0
        # Track costs and drop learning rate if cost is just bouncing around.
        if i%N1==N1-1:
            avg_costs[np.int(np.round(np.float(i%N2)/N1))-1] = np.mean(costs)
            print(np.mean(costs))
            if i%N2==N2-1:
                avg_cost = np.mean(np.sort(avg_costs)[:-2])
                print('Average cost: ' + str(avg_cost))
                recorded_costs.append(avg_cost)
                if np.mean(avg_costs)<err_best:
                    U_best, Vt_best, Y_best, err_best = U,Vt,Y,np.mean(avg_costs)
                if len(recorded_costs)>3:
                    plt.plot(range(len(recorded_costs)),recorded_costs)
                    plt.show()
                    if np.mean(recorded_costs[-3:-1])-recorded_costs[-1]<0:
                        new_eps =  max([eps*eps_drop_frac, min_eps])
                        if np.isclose(new_eps,eps):
                            print('Learning rate at minimum, ending program')
                            break
                        else:
                            print('Dropping the learning rate from {0} to {1}'.format(eps,new_eps))
                            eps = new_eps
    if implicit:
#        U_best += np.dot((data!=0)/ \
#                  (10**-8+np.dot(np.sqrt(np.sum((data_slice!=0),axis=1)), \
#                   np.ones((1,width)))),Y_best)
        return U_best, Vt_best
    else:
        return U_best, Vt_best

#def nonzero_factorization(data, d=20, batch_size=64, eps = 10**-4, svd_start = True, 
#                          verbose=True,min_eps=10**-7,eps_drop_frac=10**-0.5,
#                          lamb = 0, bias=False, implicit=False):
#    ''' Matric factorization where the zero elements are not considered'''
#    # We will be slicing out rows, so make sure data is a csr_matrix.
#    assert type(data) == scipy.sparse.csr.csr_matrix
#    height,width = data.shape
#    U_best, Vt_best, err_best = [],[],np.inf
#    if svd_start:
#        # Initialize matrix with the truncated svd which includes the zeros:
#        # This increases the likelihood of convergence.
#        U,s,Vt = scipy.sparse.linalg.svds(data,k=d)
#        s = np.diag(s)
#        U = np.dot(U,np.sqrt(s))
#        Vt = np.dot(np.sqrt(s),Vt)
#        # Add noise to let gradient descent work:
#        U[U==0] = np.random.randn(len(U[U==0]))
#        Vt[Vt==0] = np.random.randn(len(Vt[Vt==0]))
#    else:
#        # Initialize matrix with noise:
#        U = np.random.randn(height,d)
#        Vt = np.random.randn(d,width)
#    # Include a term similar to U that accounts for the fact that
#    # the user rated an item at all:
#    if implicit:
#        Y = np.random.randn(data.shape[1],d)/100
#        Fbool = np.zeros(data.shape,dtype=bool)
#        Fbool[data.nonzero()] = True
#        Frow = np.expand_dims(np.sqrt(np.sum(Fbool,axis=1)),axis=1)
#        #U -= np.dot(Fbool/(10**-8+np.dot(Frow,np.ones((1,width)))), Y)
#    # Include a user and anime bias in the results:
#    if bias:
#        U = np.append(U,np.random.random((U.shape[0],2)),axis=1)
#        Vt = np.append(Vt,np.random.random((2,Vt.shape[1])),axis=0)
#        U[:,-1] = 1
#        Vt[-2,:] = 1
#        if implicit:
#            Y = np.append(Y,np.zeros(Y.shape[0],2),axis=1)
#    # Variables to track costs:
#    N1,N2 = 100,1000
#    costs = [100]*N1
#    avg_costs = [np.Inf]*10
#    recorded_costs = []
#    for i in range(10**6):
#        # Choose random rows and slice the data:
#        rows = np.random.choice(height,batch_size)
#        data_slice = data[rows,:].todense()
#        # Determine which indices which should be used:
#        valid_entries = data_slice!=0
#        U_slice = np.copy(U[rows,:])
#        
#        # Handle calculation if implicit is True
#        if implicit:
#            # Get previous FY slice and calculate error
#            FY_slice = np.dot(Fbool[rows,:]/(10**-8+np.dot(Frow[rows,:],np.ones((1,width)))), Y)
#            error = np.multiply(np.dot(U_slice-FY_slice,Vt)-data_slice,valid_entries)
#            # Update according to that error:
#            if lamb:
#                U[rows,:] -= eps*(np.dot(error,np.transpose(Vt)) + lamb*U_slice)
#                Y-= eps*(np.dot(np.transpose(Fbool[rows,:]/(10**-8+np.dot(Frow[rows,:],np.ones((1,width))))),
#                               np.dot(error,np.transpose(Vt))) + lamb*Y)
#                Vt -= eps*(np.dot(np.transpose(U_slice+FY_slice),error)+lamb*Vt)
#                costs[i%N1] = np.sum(np.power(error,2))
#            else:
#                U[rows,:] -= eps*np.dot(error,np.transpose(Vt))
#                Y-= eps*np.dot(np.transpose(Fbool[rows,:]/(10**-8+np.dot(Frow[rows,:],np.ones((1,width))))),
#                               np.dot(error,np.transpose(Vt)))
#                Vt -= eps*np.dot(np.transpose(U_slice+FY_slice),error)
#                costs[i%N1] = np.sum(np.power(error,2))
#        else:
#            # Calculate error and update:
#            error = np.multiply(np.dot(U_slice,Vt)-data_slice,valid_entries)
#            # Handle regularization:
#            if lamb:
#                U[rows,:] -= eps*(np.dot(error,np.transpose(Vt)) + lamb*U_slice)
#                Vt -= eps*(np.dot(np.transpose(U_slice),error)+lamb*Vt)
#                costs[i%N1] = np.sum(np.power(error,2)) + \
#                               lamb/2*(np.sum(U_slice**2)+np.sum(Vt**2))
#            else:
#                U[rows,:] -= eps*np.dot(error,np.transpose(Vt))
#                Vt -= eps*np.dot(np.transpose(U_slice),error)
#                costs[i%N1] = np.sum(np.power(error,2))
#        # Reset the bias columns:
#        if bias:
#            U[:,-1] = 1
#            Vt[-2,:] = 1
#        # Track costs and drop learning rate if cost is just bouncing around.
#        if i%N1==N1-1:
#            avg_costs[np.int(np.round(np.float(i%N2)/N1))-1] = np.mean(costs)
#            print(np.mean(costs))
#            if i%N2==N2-1:
#                avg_cost = np.mean(np.sort(avg_costs)[:-2])
#                print('Average cost: ' + str(avg_cost))
#                recorded_costs.append(avg_cost)
#                if np.mean(avg_costs)<err_best:
#                    U_best, Vt_best, err_best = U,Vt,np.mean(avg_costs)
#                if len(recorded_costs)>3:
#                    plt.plot(range(len(recorded_costs)),recorded_costs)
#                    plt.show()
#                    if np.mean(recorded_costs[-3:-1])-recorded_costs[-1]<0:
#                        new_eps =  max([eps*eps_drop_frac, min_eps])
#                        if np.isclose(new_eps,eps):
#                            print('Learning rate at minimum, ending program')
#                            break
#                        else:
#                            print('Dropping the learning rate from {0} to {1}'.format(eps,new_eps))
#                            eps = new_eps
#    return U_best, Vt_best


#def nonzero_factorization(data, d=20, batch_size=64, eps = 10**-4, svd_start = True, 
#                          verbose=True,min_eps=10**-7,eps_drop_frac=10**-0.5,
#                          lamb = 0, bias=False, implicit=False):
#    # We will be slicing out rows, so make sure data is a csr_matrix.
#    assert type(data) == scipy.sparse.csr.csr_matrix
#    height,width = data.shape
#    U_best, Vt_best, err_best = [],[],np.inf
#    if svd_start:
#        # Initialize matrix with the truncated svd which includes the zeros:
#        # This increases the likelihood of convergence.
#        U,s,Vt = scipy.sparse.linalg.svds(data,k=d)
#        s = np.diag(s)
#        U = np.dot(U,np.sqrt(s))
#        Vt = np.dot(np.sqrt(s),Vt)
#        # Add noise to let gradient descent work:
#        U[U==0] = np.random.randn(len(U[U==0]))
#        Vt[Vt==0] = np.random.randn(len(Vt[Vt==0]))
#    else:
#        # Initialize matrix with noise:
#        U = np.random.randn(height,d)
#        Vt = np.random.randn(d,width)
#    if bias:
#        U = np.append(U,np.random.random((U.shape[0],2)),axis=1)
#        Vt = np.append(Vt,np.random.random((2,Vt.shape[1])),axis=0)
#        U[:,-1] = 1
#        Vt[-2,:] = 1
#    # Variables to track costs:
#    N1,N2 = 100,1000
#    costs = [100]*N1
#    avg_costs = [np.Inf]*10
#    recorded_costs = []
#    for i in range(10**6):
#        # Choose random rows and slice the data:
#        rows = np.random.choice(height,batch_size)
#        data_slice = data[rows,:].todense()
#        # Determine which indices which should be used:
#        valid_entries = data_slice!=0
#        U_slice = np.copy(U[rows,:])
#        # Calculate error and update:
#        error = np.multiply(np.dot(U_slice,Vt)-data_slice,valid_entries)
#        # Handle regularization:
#        if lamb:
#            U[rows,:] -= eps*(np.dot(error,np.transpose(Vt)) + lamb*U_slice)
#            Vt -= eps*(np.dot(np.transpose(U_slice),error)+lamb*Vt)
#            costs[i%N1] = np.sum(np.power(error,2)) + \
#                           lamb/2*(np.sum(U_slice**2)+np.sum(Vt**2))
#        else:
#            U[rows,:] -= eps*np.dot(error,np.transpose(Vt))
#            Vt -= eps*np.dot(np.transpose(U_slice),error)
#            costs[i%N1] = np.sum(np.power(error,2))
#        # Reset the bias columns:
#        if bias:
#            U[:,-1] = 1
#            Vt[-2,:] = 1
#        # Track costs and drop learning rate if cost is just bouncing around.
#        if i%N1==N1-1:
#            avg_costs[np.int(np.round(np.float(i%N2)/N1))-1] = np.mean(costs)
#            print(np.mean(costs))
#            if i%N2==N2-1:
#                avg_cost = np.mean(np.sort(avg_costs)[:-2])
#                print('Average cost: ' + str(avg_cost))
#                recorded_costs.append(avg_cost)
#                if np.mean(avg_costs)<err_best:
#                    U_best, Vt_best, err_best = U,Vt,np.mean(avg_costs)
#                if len(recorded_costs)>3:
#                    plt.plot(range(len(recorded_costs)),recorded_costs)
#                    plt.show()
#                    if np.mean(recorded_costs[-3:-1])-recorded_costs[-1]<0:
#                        new_eps =  max([eps*eps_drop_frac, min_eps])
#                        if np.isclose(new_eps,eps):
#                            print('Learning rate at minimum, ending program')
#                            break
#                        else:
#                            print('Dropping the learning rate from {0} to {1}'.format(eps,new_eps))
#                            eps = new_eps
#    return U_best, Vt_best

#def alternating_least_squares(data, d=20, iterations = 20,lamb=10**-4):
#    # Start with the simplest and least efficient algorithm.
#    # Initialize the matrices:
#    height,width = data.shape
#    U = np.random.randn(height,d)
#    Vt = np.random.randn(d,width)
#    for it in range(iterations):
#        for i in range(height):
#            for q in range(d):
#                U[i,q] = ()/(lamb+)

