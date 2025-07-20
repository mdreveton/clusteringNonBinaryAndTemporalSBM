#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:08:45 2023

@author: maximilien
"""

import numpy as np
from tqdm import tqdm
import random as random

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score


import baseFunctions as base


# =============================================================================
# MAIN CLUSTERING ALGORITHMS
# =============================================================================



def timeAggregation( adjacencyTensor, n_clusters = 2, step = 10, matrix = 'laplacian', tqdm_ = False ):
    N = adjacencyTensor.shape[ 0 ]
    T = adjacencyTensor.shape[ 2 ]
    if step > T-1:
        step = T-1
    z = np.zeros( ( N, (T-1) // step ), dtype = int )
    
    timeAggregation = np.zeros( (N,N) )
    
    if tqdm_:
        loop = tqdm( range( 1, T ) )
    else:
        loop = range( 1, T )

    for t in loop:
        timeAggregation += adjacencyTensor[:,:,t]
        if t % step == 0:
            z[ :, t//step - 1 ] = base.spectralClustering( timeAggregation, n_clusters, matrix = matrix )

    return z


def spectralConcatenation( adjacencyTensor, n_clusters, step = 10, spherical = False, embedding = 'spectral', tqdm_ = False ):
    
    N = adjacencyTensor.shape[ 0 ]
    T = adjacencyTensor.shape[ 2 ]
    if step > T-1:
        step = T-1
    z = np.zeros( ( N, (T-1) // step ), dtype = int )
    
    numberInteractions = np.zeros( ( N,N,2,2 ) )
    
    if tqdm_:
        loop = tqdm( range( 1, T ) )
    else:
        loop = range( 1, T )
        
    for t in loop:
        numberInteractions = updateNumberInteractions( numberInteractions, adjacencyTensor[:,:,t-1], adjacencyTensor[:,:,t] )
        
        if t % step == 0:
            P = getProbabilitiesFromNumberInteractions( numberInteractions )
            
            if embedding == 'spectral':
                embedding_function = spectralEmbedding
            else:
                embedding_function = lowRankEstimateExpectedAdjacencyMatrix
            
            P00 = P[ :,:,0,0 ]
            U00 = embedding_function( base.zeroingDiagonal(P00), n_clusters, spherical = spherical )

            P01 = P[ :,:,0,1 ]
            U01 = embedding_function( base.zeroingDiagonal(P01), n_clusters, spherical = spherical )
            
            P11 = P[ :,:,1,1 ]
            U11 = embedding_function( base.zeroingDiagonal(P11), n_clusters, spherical = spherical )
            
            P10 = P[ :,:,1,0 ]
            U10 = embedding_function( base.zeroingDiagonal(P10), n_clusters, spherical = spherical )
            
            U = np.concatenate( [ U00, U01, U10, U11 ], axis = 1 )
                        
            #kmeans = KMeans( n_clusters = n_clusters, init='random', n_init = 40 ).fit( U )
            kmeans = KMeans( n_clusters = n_clusters, n_init = 40 ).fit( U )
            z[ :, t//step - 1 ] = kmeans.labels_ + np.ones( N, dtype = int )
    
    return z


def spectralTimeAggregated( adjacencyTensor, n_clusters = 2, step = 10, spherical = False, embedding = 'spectral', tqdm_ = False ):
    N = adjacencyTensor.shape[ 0 ]
    T = adjacencyTensor.shape[ 2 ]
    if step > T-1:
        step = T-1
    z = np.zeros( ( N, (T-1) // step ), dtype = int )
    
    numberInteractions = np.zeros( ( N,N,2,2 ) )
    
    if embedding == 'spectral':
        embedding_function = spectralEmbedding
    else:
        embedding_function = lowRankEstimateExpectedAdjacencyMatrix

    if tqdm_:
        loop = tqdm( range( 1, T ) )
    else:
        loop = range( 1, T )

    for t in loop:
        
        numberInteractions = updateNumberInteractions( numberInteractions, adjacencyTensor[:,:,t-1], adjacencyTensor[:,:,t] )
        
        if t % step == 0:
            #P = getProbabilitiesFromNumberInteractions( numberInteractions )
            #P1 = P[ :,:,0,1 ] + P[ :,:,1,1 ]
            
            W = numberInteractions[ :,:,0,1 ] + numberInteractions[ :,:,1,1 ]
            U1 = embedding_function( W, n_clusters, spherical = spherical )
            
            #kmeans = KMeans( n_clusters = n_clusters, init='random', n_init = 40 ).fit( U1 )
            kmeans = KMeans( n_clusters = n_clusters, n_init = 40 ).fit( U1 )
            z[ :, t//step - 1 ] = kmeans.labels_ + np.ones( N, dtype = int )
    
    return z


def spectralEmbedding( W, K, spherical = False ):
    
    vals, vecs = np.linalg.eigh( W )
    U = vecs[ :, np.argsort( np.abs( vals ) )[ -K: ] ]
    Lambda = np.diag( vals[ np.argsort( np.abs( vals ) )[ -K: ] ] )
    
    if spherical == False:
        return U @ Lambda
    else:
        for i in range( W.shape[ 0 ] ):
            if np.linalg.norm( U[i,:], ord = 1) != 0:
                U[i,:] = U[i,:] / np.linalg.norm( U[i,:], ord = 1 )
        return U @ Lambda



def lowRankEstimateExpectedAdjacencyMatrix( W, K, spherical = False, threshold = 10**(-7) ):
    """
    Parameters
    ----------
    W : N-by-N matrix
        weighted adjacency matrix of a graph.

    Returns
    -------
    hatP : N-by-N matrix
          element (i,j) is the expectation of W_{ij}, assuming that \E W is of rank K.

    """    
    vals, vecs = np.linalg.eigh( W )
    hatP = vecs @ np.diag( vals ) @ vecs.T
    hatP = np.where( hatP < threshold, 0, hatP ) #some values are so small, we put them to 0

    if spherical:
        #If spherical, we normalize the rows of hatP
        for i in range( W.shape[ 0 ] ):
            if np.linalg.norm( hatP[i,:], ord = 1) != 0:
                hatP[i,:] = hatP[i,:] / np.linalg.norm( hatP[i,:], ord = 1 )

    return hatP

def iterativeSpectral( adjacencyTensor, n_clusters = 2, step = 10, n_iter = 10, epsilon = 1/1000, initialisation = 'Lasse', sc_matrix = 'adjacency' ):
    N = adjacencyTensor.shape[ 0 ]
    T = adjacencyTensor.shape[ 2 ]
    
    if step > T-1:
        step = T-1
    z = np.zeros( ( N, (T-1) // step ), dtype = int )

        
    numberInteractions = np.zeros( ( N,N,2,2 ) )
    
    timeaggregated = adjacencyTensor[ :,:,0 ]
    persistent = np.zeros( (N,N) )
    
    for t in range( 1, T ):
        numberInteractions = updateNumberInteractions( numberInteractions, adjacencyTensor[:,:,t-1], adjacencyTensor[:,:,t] )
        timeaggregated += adjacencyTensor[ :,:,t ]
        persistent += np.multiply( adjacencyTensor[ :,:,t-1 ], adjacencyTensor[ :,:,t ] )
        
        if t % step == 0:
            
            if initialisation == 'me':
                P = getProbabilitiesFromNumberInteractions( numberInteractions )
                
                labels_pred_01 = base.spectralClustering_adjacencyMatrix( P[:,:,0,1], n_clusters = n_clusters ) 
                labels_pred_11 = base.spectralClustering_adjacencyMatrix( P[:,:,1,1], n_clusters = n_clusters )
                
                w01_in, w01_out = base.computeMeans( P[:,:,0,1], labels_pred_01 )
                w11_in, w11_out = base.computeMeans( P[:,:,1,1], labels_pred_01 )
                
                I01 = ( w01_in - w01_out )**2 #/ ( w01_in + (n_clusters - 1) * w01_out )
                I11 = ( w11_in - w11_out )**2 #/ ( w11_in + (n_clusters - 1) * w11_out )
        
                if I01 > I11 :
                    z_ = labels_pred_01
                    w11_in, w11_out = base.computeMeans( P[:,:,1,1], z_ )
                else:
                    z_ = labels_pred_11
                    w01_in, w01_out = base.computeMeans( P[:,:,0,1], z_ )
            
            elif initialisation == 'Lasse':
                P = getProbabilitiesFromNumberInteractions( numberInteractions )
                
                P01 = P[ :,:,0,1 ]
                vals, vecs = np.linalg.eigh( P01 )
                U01 = vecs[ :, np.argsort( np.abs( vals ) )[-n_clusters:] ]
                
                P11 = P[ :,:,1,1 ]
                vals, vecs = np.linalg.eigh( P11 )
                U11 = vecs[ :, np.argsort( np.abs( vals ) )[-n_clusters:] ]
                
                U = np.concatenate([U01, U11], axis = 1 )
                kmeans = KMeans( n_clusters = n_clusters, n_init="auto" ).fit( U )
                z_= kmeans.labels_ + np.ones( N, dtype = int )
            
            
            iteration = 0
            refinement = ( iteration < n_iter )
            z_previous = z_
            
            while refinement:
                w01_in, w01_out = base.computeMeans( P[:,:,0,1], z_previous )
                w11_in, w11_out = base.computeMeans( P[:,:,1,1], z_previous )

                if w01_out == 0:
                    w01_out = epsilon
                if w11_out == 0:
                    w11_out = epsilon
                if w11_in == 0:
                    w11_in = epsilon
                if w01_in == 0:
                    w01_in = epsilon

                alpha = np.log( ( w01_in ) / ( w01_out ) ) +  np.log( ( 1 - w11_in ) / ( 1 - w11_out ) )
                beta = np.log( ( w11_in ) / ( w11_out ) )
                
                W = alpha * timeaggregated + ( beta - alpha ) * persistent
                
                z_refined = base.spectralClustering( W, n_clusters = n_clusters, sc_matrix = sc_matrix )
                
                iteration += 1
                if iteration >= n_iter or adjusted_rand_score( z_refined, z_previous ) > 0.98:
                    refinement = False
                    #print( 'The number of refinement step done is : ', iteration )
                z_previous = z_refined
            
            z[ :, t//step - 1 ] = z_refined
    
    return z



# =============================================================================
# ADDITIONAL FUNCTIONS
# =============================================================================


def computeNumberOfInteractions( adjacencyTensor ):
    N = adjacencyTensor.shape[ 0 ]
    T = adjacencyTensor.shape[ 2 ]
    numberInteractions = np.zeros( ( N,N,2,2 ) )
    
    for t in range( 1, T ):
        numberInteractions = updateNumberInteractions( numberInteractions, adjacencyTensor[:,:,t-1], adjacencyTensor[:,:,t] )
    
    return numberInteractions



def updateNumberInteractions( numberInteractions, A_previous, A_new ): 
    
    numberInteractions[:,:, 0, 0 ] += np.multiply( np.where( A_previous == 0, 1, 0), np.where( A_new == 0, 1, 0 ) )
    numberInteractions[:,:, 0, 1 ] += np.multiply( np.where( A_previous == 0, 1, 0), A_new )
    numberInteractions[:,:, 1, 0 ] += np.multiply( A_previous, np.where( A_new == 0, 1, 0 ) )
    numberInteractions[:,:, 1, 1 ] += np.multiply( A_previous, A_new )
    
    return numberInteractions


def getProbabilitiesFromNumberInteractions( numberInteractions ):
    P = np.zeros( numberInteractions.shape )
    
    N00 = numberInteractions[:,:,0,0]
    N01 = numberInteractions[:,:,0,1]
    N0 = N00 + N01 + 0.1 * np.ones( (P.shape[0], P.shape[1]) )
    P[:,:,0,0] = np.divide( N00, N0, where = (N0 > 0) )
    P[:,:,0,1] = np.divide( N01, N0, where = (N0 > 0) )
    #P[:,:,0,0] = np.where( N0 != 0, numberInteractions[:,:,0,0] / N0, 0.5 )
    #P[:,:,0,1] = np.where( N0 != 0, numberInteractions[:,:,0,1] / N0, 0.5 )
    
    N10 = numberInteractions[:,:,1,0]
    N11 = numberInteractions[:,:,1,1]
    N1 = N10 + N11 + 0.1 * np.ones( (P.shape[0], P.shape[1]) )
    P[:,:,1,0] = np.divide( N10, N1, where = (N1 > 0) )
    P[:,:,1,1] = np.divide( N11, N1, where = (N1 > 0) )
    #P[:,:,1,0] = np.where( N1 != 0, numberInteractions[:,:,1,0] / N1, 0.99 )
    #P[:,:,1,1] = np.where( N1 != 0, numberInteractions[:,:,1,1] / N1, 0.01 )
    
    return P
