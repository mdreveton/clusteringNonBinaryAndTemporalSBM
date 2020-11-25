#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:07:59 2020

@author: mdreveto
"""

import numpy as np
from tqdm import tqdm
import random as random

import baseline_clustering_algorithms as baseline_algo



def likelihoodClusteringUnkownParameters( adjacencyMatrix, initialisation = "SpectralClustering", K = 2, useTqdm = False ):
    """
    Cluster algo of the paper when the parameters are unknown.
        Step t=1 : use SpectralClustering or RandomGuessing as initialisation
        Then at step t+1, update the prediction using the maximum likelihood estimator

    Input: 
        adjacencyMatrix: n x n x T matrix 
        EdgeRateMatrix : K x K matrix (K = number of communities), whose elements (k,l) is p_{kl} (proba edge between node in community k and node in community l)
        TemporalEdgeEvolutionMatrix  : K x K matrix whose element kl is the Markovian parameter r_{kl}
        
        initialisation: the initialisation method (only SpectralClustering and RandomGuessing are implemented)

    Output: labelsPred : a matrix n x T, whose slice labelsPred[:,t] gives the labelling at time t
    """
    
    n = adjacencyMatrix.shape[0]
    T = adjacencyMatrix.shape[2]
    labelsPred = np.zeros( [n,T] )
    if(initialisation == "SpectralClustering"):
        labelsPred[:,0] = baseline_algo.staticSpectralClustering( adjacencyMatrix[:,:,0] ) #Spectral Clustering prediction
    elif(initialisation == "RandomGuessing"):
        for i in range(n):
            labelsPred[i,0] = ( random.random() < 0.5 ) * 1
    else:
        labelsPred[:,0] = initialisation
        #return print("Propose a correct initialisation procedure")
    labelsPred.astype(int)



    likelihood = np.zeros( ( n, K ) )
    transitionsMatrix = np.zeros( (n, n, 2, 2)  ) #This will store, at time t, the number of transitions of type (a,b) between nodes i and j (a, b can be 0 or 1)
    
    if useTqdm:
        loop = tqdm(range(1, T) )
    else:
        loop = range(1, T)
    
    for t in loop:
        
        for i in range(n):
            for j in range(i):
                transitionsMatrix[ i, j, adjacencyMatrix[i,j,t-1], adjacencyMatrix[i,j,t] ] += 1
                transitionsMatrix[ j, i, adjacencyMatrix[j,i,t-1], adjacencyMatrix[j,i,t] ] += 1
        
        nodesInEachCluster = []
        for cluster in range(K):
            nodesInEachCluster.append( [ i for i in range(n) if labelsPred[i,t-1] == cluster ] )
        
        #We need estimators for the parameters Pin and Pout (Markov transition probabilities) as well as piin and piout (initial distribution)
        (piin, piout, Pin, Pout) = parameterEstimation(adjacencyMatrix[:,:,0], transitionsMatrix, nodesInEachCluster, n , K = K)
        
        #Now that we have estimates for Pin and Pout, we can compute the likelihood
        l = np.zeros( (2,2) )
        l[0,0] = np.log( Pin[0,0] / Pout[0,0] )
        if( Pin[0,1] * Pout[0,1] != 0 ):
            l[0,1] = np.log( Pin[0,1] / Pout[0,1] )
        if(Pin[1,0] * Pout[1,0] != 0):
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
        ell[0] = np.log( (1-piin) / (1-piout) )
        ell[1] = np.log ( piin / piout )
        
        M = np.zeros( (n,n) )

        for i in range(n):
            for j in range(i):
                M[i,j] += ell[ int(adjacencyMatrix[i,j,0]) ] 
                for tprime in range(1,t):
                    M[i,j] += l[ adjacencyMatrix[i,j,tprime-1], adjacencyMatrix[i,j,tprime] ]
                M[j,i] = M[ i,j ]
        
        nodeOrder = [i for i in range(n)] 
        random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
        for i in nodeOrder:
            for cluster in range(K):
                likelihood[ i,cluster ]  = 0
                for node in nodesInEachCluster[cluster]:
                    likelihood[i,cluster] += M[ i,node ]
            labelsPred[i,t] = np.argmax( likelihood[i,:] )

    return (labelsPred, Pin, Pout)





def parameterEstimation( initial_adjacency_matrix, transitionsMatrix, nodesInEachCluster, n , K = 2 ):
    """
    Return estimators for the parameters Pin and Pout (Markov transition probabilities) as well as piin and piout (initial distribution) 
    given a clustering (nodesInEachCluster) (this can be an estimated clustering)
    """
    Pin = np.zeros( (2,2) )
    Pout = np.zeros( (2,2) )
    piin = 0
    piout = 0
    count = 0
    n0 = 0
    n1 = 0
    for cluster in range(K):
        for i in nodesInEachCluster[cluster]:
            for j in nodesInEachCluster[cluster]:
                if (j!=i):
                    n1 += transitionsMatrix[ i, j, 1, 0 ] + transitionsMatrix[ i, j, 1, 1 ]
                    n0 += transitionsMatrix[ i, j, 0, 0 ] + transitionsMatrix[ i, j, 0, 1 ]
                    
                    Pin[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                    Pin[ 0,0 ] += transitionsMatrix[ i, j, 0, 0] 
                    Pin[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                    Pin[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                    
                    piin += initial_adjacency_matrix[i , j]
                    count += 1
    #print(Pin)
    if(count != 0):
        piin = piin / count
    if( n0!=0 ):
        #Pin[1,0] /= n0
        Pin[0,1] /= n0
        Pin[0,0] /= n0
    if (n1 != 0 ):
        #Pin[0,1] /= n1
        Pin[1,0] /= n1
        Pin[1,1] /= n1
    
    count = 0
    n0 = 0
    n1 = 0
    for cluster in range(K):
        for i in nodesInEachCluster[cluster]:
            otherClustersNodes = [dummy for dummy in range(n) if dummy not in nodesInEachCluster[cluster] ]
            for j in otherClustersNodes:
                n1 +=  transitionsMatrix[ i, j, 1, 0 ] + transitionsMatrix[ i, j, 1, 1 ]
                n0 += transitionsMatrix[ i, j, 0, 0 ] + transitionsMatrix[ i, j, 0, 1 ]
                
                Pout[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                Pout[ 0,0 ] += transitionsMatrix[ i, j, 0, 0] 
                
                Pout[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                Pout[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                
                piout += initial_adjacency_matrix[ i, j]
                count += 1
    #print(Pout)
    if (count != 0):
        piout = piout / count
    if( n0!=0 ):
        Pout[0,1] /= n0
        Pout[0,0] /= n0
    if (n1 != 0 ):
        Pout[1,0] /= n1
        Pout[1,1] /= n1
    
    return (piin, piout, Pin, Pout)





