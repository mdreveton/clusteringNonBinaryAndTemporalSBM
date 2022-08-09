#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 17:10:24 2022

@author: maximilien
"""

import numpy as np
import spectralClustering as sc
from tqdm import tqdm 



def twoStepAlgo_MarkovSBM( adjacencyTensor, mu, nu, P, Q, K = 2, tqdm_ = False, improvementStep = True ):
    
    N = adjacencyTensor.shape[ 0 ]

    loglikelihoods = markovLogLikelihood( adjacencyTensor, mu, nu, P, Q, epsilon = 0 )
    
    T = adjacencyTensor.shape[ 2 ]
    likelihoodOfNoInteractions = np.log( markovPathLikelihood( np.zeros( T, dtype = int ), mu, P ) / markovPathLikelihood( np.zeros( T, dtype = int ), nu, Q ) )
    if likelihoodOfNoInteractions < 0:
        A_forInitialisation = np.where( loglikelihoods > likelihoodOfNoInteractions, 1, 0 )
    else:
        A_forInitialisation = np.where( loglikelihoods >= likelihoodOfNoInteractions, 1, 0 )
        #A_forInitialisation = np.where( loglikelihoods > likelihoodOfNoInteractions, 1, 0 )

    
    sigma_tilde = sc.staticSpectralClustering( A_forInitialisation, K = K, assign_labels = 'discretize' )
    
    if improvementStep == False:
        return sigma_tilde
    else:
        sigma = sigma_tilde.copy( )
        
        if tqdm_:
            loop = tqdm( range( N ) )
        else:
            loop = range( N )
        
        for i in loop:
            likelihood_prediction_for_node_i = likelihood_test( i, sigma_tilde, loglikelihoods, K = K )
            sigma[ i ] = likelihood_prediction_for_node_i
        
        return sigma


def markovLogLikelihood( adjacencyTensor, mu, nu, P, Q, epsilon = 0 ):
    N = adjacencyTensor.shape[ 0 ]
    W = np.zeros( ( N, N ) )
    for i in range( N ):
        for j in range( i ):
            x = adjacencyTensor[ i, j, : ]
            W[ i, j ] = np.log( (epsilon + markovPathLikelihood( x, mu, P ) ) / ( epsilon + markovPathLikelihood( x, nu, Q ) ) )
            W[ j, i ] = W[ i, j ]
    return W


def markovPathLikelihood( x, mu, P ):
    likelihood = mu[ x[0] ]
    for t in range( 1, len( x ) ):
        likelihood += P[ x[t-1], x[t] ]
    return likelihood


def likelihood_test( node, labels_pred, loglikelihoods, K = 2 ):
    n = len( labels_pred )
    loglikelihoods_of_given_node = np.zeros( K )
    for i in range( n ):
        loglikelihoods_of_given_node[ labels_pred[i]-1 ] += loglikelihoods[ node, i ]
                
    return np.argmax( loglikelihoods_of_given_node ) + 1
