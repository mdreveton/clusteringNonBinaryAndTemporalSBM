#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:25:18 2022

@author: maximilien
"""

import numpy as np
import scipy as sp
from tqdm import tqdm
from sklearn.cluster import KMeans, SpectralClustering

import sklearn.cluster as cluster


def staticSpectralClustering( adjacencyMatrix, K = 2, assign_labels = 'kmeans' ):
    
    sc = SpectralClustering( n_clusters = K, affinity = 'precomputed', assign_labels = assign_labels )
    labels_pred_spec = sc.fit_predict( adjacencyMatrix ) + np.ones( adjacencyMatrix.shape[0] )
    
    return labels_pred_spec.astype( int )


def meanAdjacencyMatrixSpectralClustering( adjacencyTensor, K = 2, useTqdm = False, timestep = 1, assign_labels = 'kmeans' ):
    T = adjacencyTensor.shape[2]
    N = adjacencyTensor.shape[0]
    labels_pred = np.ones( ( N, T // timestep +1 ), dtype = int )
    
    if useTqdm:
        loop = tqdm( range( 0, T ) )
    else:
        loop = range( 0, T )
    
    adjacencyMatrixAggregated = np.zeros( ( N, N ) )
    
    for t in loop:
        adjacencyMatrixAggregated[ :, : ] += adjacencyTensor[ :, :, t ]
        if t % timestep == 0:
            vals, vecs = np.linalg.eigh( adjacencyMatrixAggregated )
            vecs = vecs[ :, -K: ]
            #vals, vecs = sp.sparse.linalg.eigsh( sp.sparse.csr_matrix( adjacencyMatrixAggregated ), k = K, which = 'LM' )
            if assign_labels == 'kmeans':
                kmeans = KMeans( n_clusters = K ).fit( vecs )
                labels_pred[ :, t // timestep ] = kmeans.labels_ + np.ones( N )
            
            elif assign_labels == 'discretize':
                labels_pred[ :, t // timestep ] = cluster._spectral.discretize( vecs ) + np.ones( N )
    
    return labels_pred.astype( int )


def sumOfSquaredSpectralClustering( adjacencyTensor, K = 2, biais_adjusted = True , useTqdm = False, timestep = 1, assign_labels = 'kmeans' ):
    """
    Corresponds to the algo studied in the paper:
        Jing  Lei. Tail  bounds  for  matrix  quadratic  forms  and bias  adjusted  spectral  clustering  in  multi-layer  stochastic  block  models.
    """
    
    T = adjacencyTensor.shape[ 2 ]
    N = adjacencyTensor.shape[ 0 ]
    labels_pred = np.zeros( ( N, T // timestep + 1 ), dtype = int)
    
    if useTqdm:
        loop = tqdm( range( T ) )
    else:
        loop = range( T )
    
    adjaSquaredSum = sp.sparse.csr_matrix( adjacencyTensor[ :, :, 0 ] ) @  sp.sparse.csr_matrix( adjacencyTensor[ :, :, 0 ] )
    if biais_adjusted:
        adjaSquaredSum -= sp.sparse.diags ( adjaSquaredSum.diagonal( ) )

    for t in loop:
        newSquare = sp.sparse.csr_matrix( adjacencyTensor[ :, :, t ] ) @ sp.sparse.csr_matrix( adjacencyTensor[ :, :, t ] )
        if biais_adjusted:
            adjaSquaredSum += newSquare - sp.sparse.diags ( newSquare.diagonal( ) )
        else:
            adjaSquaredSum += newSquare
        
        if t % timestep == 0:
            vals, vecs = sp.sparse.linalg.eigsh( adjaSquaredSum.asfptype( ) , k = K, which = 'LM' )
        
            if assign_labels == 'kmeans':
                kmeans = KMeans( n_clusters = K ).fit( vecs )
                labels_pred[ :, t // timestep ] = kmeans.labels_ + np.ones( N )
        
            elif assign_labels == 'discretize':
                labels_pred[ :, t// timestep ] = cluster._spectral.discretize( vecs ) + np.ones( N )
    
    return labels_pred.astype( int )




def timeAggregatedSpectralClustering( adjacencyTensor , K = 2, useTqdm = False, timestep = 1, assign_labels = 'kmeans' ):
    
    T = adjacencyTensor.shape[2]
    N = adjacencyTensor.shape[0]
    labels_pred = np.zeros( ( N, T // timestep + 1 ), dtype = int )
    
    if useTqdm:
        loop = tqdm( range( 0, T ) )
    else:
        loop = range( 0, T )
    
    adjacencyMatrixAggregated = np.zeros( ( N, N ) )
    
    for t in loop:
        adjacencyMatrixAggregated[ :, : ] += adjacencyTensor[ :, :, t ]
        if t % timestep == 0:
            sc = SpectralClustering( n_clusters = K, affinity = 'precomputed', assign_labels = assign_labels )
            labels_pred[ :, t // timestep ] = sc.fit_predict( adjacencyMatrixAggregated ) + np.ones( N )
    
    return labels_pred.astype( int )

