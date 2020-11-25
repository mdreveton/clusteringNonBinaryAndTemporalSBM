#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Tue Aug 25 14:55:26 2020

@author: Maximilien Dreveton

"""

import numpy as np
from tqdm import tqdm
import random as random
from sklearn.metrics import accuracy_score

import baseline_clustering_algorithms as baseline_algo



def likelihoodClustering(adjacencyMatrix, initialDistributionRateMatrix, TransitionRateMatrix, initialisation = "SpectralClustering", method = "continuous_update", K = 2, useTqdm = False):
    """
    Cluster algo of the paper.
        Step t=1 : use SpectralClustering or RandomGuessing as initialisation
        Then at step t+1, update the prediction using the maximum likelihood estimator
        
    Input: 
        adjacencyMatrix: n x n x T matrix 
        initialDistributionRateMatrix : K x K matrix (K = number of communities), whose elements (k,l) is p_{kl} (proba edge between node in community k and node in community l).
        TransitionRateMatrix : a K x K matrix whose element kl is a 2-by-2 matrix ( it is the Markov Transition matrix P_{kl} )
        
        initialisation: the initialisation method (only SpectralClustering and RandomGuessing are implemented)
        
    Output: labelsPred : a matrix n x T, whose slice labelsPred[:,t] gives the labelling at time t
    """
    
    n = adjacencyMatrix.shape[0]
    T = adjacencyMatrix.shape[2]
    labelsPred = np.zeros([n,T], dtype=int )
    if( initialisation == "SpectralClustering" ):
        labelsPred[:,0] = baseline_algo.staticSpectralClustering( adjacencyMatrix[:,:,0] ) #Spectral Clustering prediction
    elif( initialisation == "RandomGuessing" ):
        for i in range(n):
            labelsPred[i,0] = (random.random() < 0.5)*1
    else:
        return print("Propose a correct initialisation procedure")

    likelihood = np.zeros((n,2))
    pin = initialDistributionRateMatrix[0,0]
    pout = initialDistributionRateMatrix[0,1]
    Pin = TransitionRateMatrix[0,0,:]
    Pout = TransitionRateMatrix[0,1,:]
    
    l = np.zeros((2,2))
    l[0,0] = np.log( Pin[0,0] / Pout[0,0] )
    if( Pin[0,1] * Pout[0,1] !=0 ):
        l[0,1] = np.log( Pin[0,1] / Pout[0,1] )
    if(Pin[1,0] * Pout[1,0] !=0):
        l[1,0] = np.log( Pin[1,0] / Pout[1,0] )
    if (Pin[1,1] * Pout[1,1] != 0):
        l[1,1] = np.log( Pin[1,1] / Pout[1,1] )
    
    if Pin[0,1] == 0:
        l[0,1] = - 9999
    if Pout[0,1] == 0:
        l[0,1] = + 9999
    if (Pin [0,1] == 0 and Pout[0,1] == 0):
        l[0,1] = 0
    if Pin[1,0] == 0:
        l[1,0] = - 9999
    if Pout[1,0] == 0:
        l[1,0] = + 9999
    if (Pin [1,0] == 0 and Pout[1,0] == 0):
        l[1,0] = 0

    
    ell = np.zeros(2)
    ell[0] = np.log((1-pin) / (1-pout))
    ell[1] = np.log (pin / pout)
    
    if useTqdm:
        loop = tqdm(range(1, T) )
    else:
        loop = range(1, T)

    
    if(method == "likelihoodMatrix"): 
        M = np.zeros((n,n))
        for i in range(n):
            for j in range(i-1):
                M[i, j] = ell[ int(adjacencyMatrix[i,j,0]) ]
                M[j,i] = M[i,j]
        for t in loop:
            for i in range(n):
                for j in range(i):
                    M[i,j] += l[ int(adjacencyMatrix[i,j,t-1]), int(adjacencyMatrix[i,j,t])  ]
                    M[j,i] = M[i,j]
                    
            nodesInEachCluster = []
            nodesInEachCluster.append( [ i for i in range(n) if labelsPred[i,t-1] == 0 ])
            nodesInEachCluster.append( [ i for i in range(n) if labelsPred[i,t-1] == 1 ])
            
            nodeOrder = [i for i in range(n)]
            random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
            for i in nodeOrder:
                for cluster in range(K):
                    likelihood[i,cluster]  = 0
                    for node in nodesInEachCluster[cluster]:
                        likelihood[i,cluster] += M[i,node]
                labelsPred[i,t] = np.argmax( likelihood[i,:] )
                
    elif method == "continuous_update":
        """
        This is a method where, at each time step, in clustering node i+1, we use a cluster assignment taking into account the update of nodes 1 to i.
        """
        M = np.zeros((n,n))
        for i in range(n):
            for j in range(i-1):
                M[i, j] = ell[ int(adjacencyMatrix[i,j,0]) ]
                M[j,i] = M[i,j]
        
        for t in loop:
            for i in range(n):
                for j in range(i):
                    M[i,j] += l[ int(adjacencyMatrix[i,j,t-1]), int(adjacencyMatrix[i,j,t])  ]
                    M[j,i] = M[i,j]
                    
            nodesInEachCluster = []
            nodesInEachCluster.append( [ i for i in range(n) if labelsPred[i,t-1] == 0 ] )
            nodesInEachCluster.append( [ i for i in range(n) if labelsPred[i,t-1] == 1 ] )
            
            nodeOrder = [i for i in range(n)]
            random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
            for i in nodeOrder:
                for cluster in range(K):
                    likelihood[i,cluster]  = 0
                    for node in nodesInEachCluster[cluster]:
                        likelihood[i,cluster] += M[i,node]
                labelsPred[i,t] = np.argmax( likelihood[i,:] )
                if labelsPred[i,t] != labelsPred[i, t-1]:
                    nodesInEachCluster[ labelsPred[i,t] ].append(i)
                    nodesInEachCluster[ labelsPred[i, t-1] ].remove(i)
                    
    else:
        return print("Propose a correct likelihood computation method")

    return labelsPred




def followAccuracy(nodesLabels, labelsPred):
    """
    This function compute the accuracy of the labelling labelsPred at each timestep t
    """
    accuracy = []
    for t in range(labelsPred.shape[1]):
        accuracy.append( max( accuracy_score(nodesLabels, labelsPred[:,t] ) , 1 - accuracy_score(nodesLabels, labelsPred[:,t] ) ) )
    return accuracy