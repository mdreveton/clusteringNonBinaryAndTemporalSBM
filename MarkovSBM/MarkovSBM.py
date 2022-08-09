#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:40:59 2020

@author: mdreveto
"""

import numpy as np
import networkx as nx

import scipy as sp
from scipy.sparse.linalg import eigsh
from scipy import sparse
from tqdm import tqdm
import random as random
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


"""

# =============================================================================
# Example of generating a Markov SBM
# =============================================================================

N = 500
T = 10
muin = 0.02
muout = 0.01

(qin, qout) = ( 0.35, 0.3 ) #Those are the intra and inter link persistence, ie the probabilities of transition 1  \to 1 (number between 0 and 1, close to 1 means high link persistence across time, close to zero means spikes)

Pin = makeTransitionMatrix( [1-muin, muin], qin )
Pout = makeTransitionMatrix( [1-muout, muout], qout )

TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )
initialDistributionRateMatrix = np.array( [ [muin, muout] , [muout, muin] ] )


nodesLabels = np.ones( N, dtype = int )
for i in range( N ):
    nodesLabels[i] = ( np.random.rand() < 0.5 ) *1

MSSBM_adja = makeMDSBMAdjacencyMatrix( N, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels )


"""


def makeMarkovSBMtemporalEdges( N, T, initialDistributionRateMatrix, transitionRateMatrix, nodesLabels , tqdm_ = False ):
    temporalEdges = dict( )

    for t in range( T ):
        temporalEdges[ t ] = [ ]
        
    if tqdm_:
        loop = tqdm( range( N ) )
    else:
        loop = range( N )
        
    for i in loop:
        for j in range( i, N ):
            interactionTimes = generateInteractionTimes( T, initialDistributionRateMatrix[ nodesLabels[ i ] - 1, nodesLabels[ j ] - 1 ], transitionRateMatrix[ nodesLabels[ i ] -1, nodesLabels[ j ]-1 ] )
            for interaction in interactionTimes:
                temporalEdges[ interaction ].append( ( i, j ) )
    
    return temporalEdges


def generateInteractionTimes( T, initialLinkProbability, TransitionMatrix ):
    
    interactionTimes = [ ]
    x = np.zeros( T )
    x[0] = ( random.random() < initialLinkProbability )*1
    
    for t in range( 1, T ):
        if x[ t-1 ] == 0:
            x[ t ] = ( random.random() < TransitionMatrix[0,1] ) * 1 #Proba of jump from 0 to 1
        else:
            interactionTimes.append( t-1 )
            x[ t ] = (random.random() < TransitionMatrix[1,1] ) * 1 #proba of stay from 1 to 1
    
    return interactionTimes



def makeMDSBMAdjacencyMatrix( N, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels , tqdm_ = False):
    """
    Return the adjacency matrix of a Markov SBM graph:
        An edge (i,j) is a time serie whose evolution follow a Markov Chain,
            whose initial distribution is initialDistributionRateMatrix[nodesLabels[i], nodesLabelsbels[j]]
            and the transitions matrix is TransitionRateMatrix[nodesLabels[i], nodesLabelsbels[j]]
    
    Inputs:
    N : number of nodes
    T : number of snapshots
    initialDistributionRateMatrix : K x K matrix (K = number of communities), whose elements (k,l) is p_{kl} (proba edge between node in community k and node in community l). It is not necessary to take the initial distribution being the stationnary distribution of the Markov Chain (even though we did it during all of our experiments)
    TransitionRateMatrix  : K x K matrix whose element kl is the Markov Transition matrix P_{kl}
    nodesLabels : n x 1 array whose element i gives the community of node i (ground truth)
    
    Output: adjacencyMatrix (n x n x T matrix)
    """
    adjacencyMatrix = np.zeros( ( N, N, T ), dtype=int )
    if tqdm_:
        loop = tqdm( range(N) )
    else:
        loop = range( N )
    for i in loop:
        for j in range(i):
            x = makeTimeSerie( T, initialDistributionRateMatrix[ nodesLabels[i] - 1, nodesLabels[j] - 1 ] , TransitionRateMatrix[ nodesLabels[i] - 1, nodesLabels[j] - 1 ] )
            adjacencyMatrix[ i, j, : ] = x
            adjacencyMatrix[ j, i, : ] = x
    return adjacencyMatrix


def makeTimeSerie( T, initialDistribution, TransitionMatrix ):
    x = np.zeros( T )
    x[0] = ( random.random() < initialDistribution )*1
    for i in range( 1, T ):
        if x[i-1] == 0:
            x[i] = ( random.random() < TransitionMatrix[0,1] ) * 1 #Proba of jump from 0 to 1
        else:
            x[i] = (random.random() < TransitionMatrix[1,1] ) * 1 #proba of stay from 1 to 1
    return x

