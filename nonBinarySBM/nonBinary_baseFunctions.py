#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:57:07 2023

@author: maximilien
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans


def spectralClustering( A, n_clusters = 2, trimming = False ):
    """
    Spectral clustering using the adjacency matrix
    """
    
    vals, vecs = np.linalg.eigh( A )
    U = vecs[ :, np.argsort( np.abs( vals ) )[-n_clusters:] ]
    Lambda = np.diag( vals[ np.argsort( np.abs( vals ) )[-n_clusters:] ] )   
    U = U @ Lambda
    kmeans = KMeans( n_clusters = n_clusters, n_init = "auto" ).fit( U )
    
    z_pred = kmeans.labels_ + np.ones( A.shape[ 0 ], dtype = int )

    return z_pred.astype( int )    


def oneHotRepresentation( z, n_clusters = 2):
    
    if n_clusters is None:
        n_clusters = len( set(z) )
    
    if len( set( z ) ) > n_clusters:
        raise TypeError( 'There is a mistake in the number of clusters for a one hot representation')
    
    n = len( z )
    Z = np.zeros( ( n, n_clusters ), dtype = int )
    for i in range( n ):
        Z[ i, z[i]-1 ] = 1
    
    return Z
    

def linkProbabilityEstimation( A, z ):
    """
    Estimates the probability p, q of a homogeneous SBM (given by its adjacency matrix A)
    with the community labelling z
    
    THIS COULD BE SPEED-UP A LOT, going from complexity N^2 to M. 
    """
    N = A.shape[ 0 ]
    p, q, nsame, ndiff = 0, 0, 0, 0
    
    for i in range( N ):
        for j in range( i ):
            if z[ i ] == z[ j ]:
                p += ( A[ i, j ] != 0) *1
                nsame += 1
            else:
                q += ( A[ i, j ] != 0) *1
                ndiff += 1
    
    if nsame !=0:
        p /= nsame
    else:
        p = 0
    if ndiff != 0:
        q /= ndiff
    else:
        q = 0
    
    return p, q



def nodeLikelihoods_edgeLabeled_SBM( node, A, P, Q, z, n_clusters = 2, nodesInEachCommunity = None ):
    
    likelihoods = np.zeros( n_clusters )
    
    if nodesInEachCommunity is None:
        nodesInEachCommunity = [ ]
        for k in range( n_clusters ):
            nodesInEachCommunity.append ( [ j for j in range( A.shape[ 0 ] ) if z[j] == k+1 ] )
            
    for k in range( len( nodesInEachCommunity ) ):    
        for j in nodesInEachCommunity[ k ]:
            if j != node:
                likelihoods[ k ] += np.log( P[ A[ node, j ] ] / Q[ A[ node, j ] ] )
    
    return likelihoods