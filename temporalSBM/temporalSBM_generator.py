#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:14:07 2023

@author: maximilien
"""

import numpy as np
import networkx as nx
import random as random
from tqdm import tqdm



def generate_markovSBM( T, initialDistributionRateMatrix, TransitionRateMatrix, communityLabels, tqdm_ = False ):
    
    N = len( communityLabels )
    adjacencyTensor = np.zeros( (N,N,T) , dtype = int )
    
    if tqdm_:
        loop = tqdm( range( N ) )
    else:
        loop = range( N )
        
    for i in loop:
        for j in range( i ):
            if communityLabels[i] == communityLabels[j]:
                initialDistribution_ij = initialDistributionRateMatrix[ 'intra' ]
                transitionMatrix_ij = TransitionRateMatrix[ 'intra' ]
            else:
                initialDistribution_ij = initialDistributionRateMatrix[ 'inter' ]
                transitionMatrix_ij = TransitionRateMatrix[ 'inter' ]

            adjacencyTensor[i,j,:] = generateTimeSerie( T, initialDistribution_ij, transitionMatrix_ij )
            adjacencyTensor[ j,i,: ] = adjacencyTensor[ i, j, : ]
    
    return adjacencyTensor


def generate_sparseMarkovSBM( sizes, p, T, initialDistributionRateMatrix, TransitionRateMatrix, tqdm_ = False ):
    N = sum( sizes) 
    adjacencyTensor = np.zeros( (N,N,T) , dtype = int )    
    K = len( sizes )
    
    communityLabels = [ ]
    for k in range( K ):
        communityLabels += [ k+1 for i in range( sizes[ k ] ) ]
    
    G = nx.stochastic_block_model( sizes, p )

    for edge in G.edges():
        i = edge[ 0 ]
        j = edge[ 1 ]
        if communityLabels[i] == communityLabels[j]:
            initialDistribution_ij = initialDistributionRateMatrix[ 'intra' ]
            transitionMatrix_ij = TransitionRateMatrix[ 'intra' ]
        else:
            initialDistribution_ij = initialDistributionRateMatrix[ 'inter' ]
            transitionMatrix_ij = TransitionRateMatrix[ 'inter' ]

        adjacencyTensor[i,j,:] = generateTimeSerie( T, initialDistribution_ij, transitionMatrix_ij )
        adjacencyTensor[ j,i,: ] = adjacencyTensor[ i, j, : ]
    
    return adjacencyTensor


def generateTimeSerie( T, initialDistribution, TransitionMatrix ):
    x = np.zeros( T )
    x[0] = ( random.random() < initialDistribution[1] ) * 1
    for i in range( 1, T ):
        if x[i-1] == 0:
            x[i] = ( random.random() < TransitionMatrix[0,1] ) * 1 #Proba of jump from 0 to 1
        else:
            x[i] = ( random.random() < TransitionMatrix[1,1] ) * 1 #Proba of stay from 1 to 1
    return x


def computeTransitionMatrix( stationnaryDistribution, linkPersistence ):
    p = stationnaryDistribution[1]
    P = np.zeros( (2,2) )
    P[1,1] = linkPersistence
    P[1,0] = 1 - linkPersistence
    P[0,1] = p * ( 1-linkPersistence) / (1-p)
    P[0,0] = 1 - P[0,1]
    return P
