#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:23:11 2023

@author: dreveton
"""


import numpy as np
from sklearn.cluster import SpectralClustering, KMeans



# =============================================================================
# Basic clustering algorithms
# =============================================================================


def spectralClustering( adjacencyMatrix, n_clusters = 2, matrix = 'laplacian' ):
    
    if matrix == 'laplacian':
        return spectralClustering_normalisedLaplacian( adjacencyMatrix, n_clusters = n_clusters )
    elif matrix == 'adjacency':
        return spectralClustering_adjacencyMatrix( adjacencyMatrix, n_clusters = n_clusters )
    else:
        raise TypeError( 'This spectral clustering choice is not implemented' )


def spectralClustering_normalisedLaplacian( adjacencyMatrix, n_clusters, assign_labels = 'kmeans' ):
    
    sc = SpectralClustering( n_clusters = n_clusters, affinity = 'precomputed', assign_labels = assign_labels )
    labels_pred_spec = sc.fit_predict( adjacencyMatrix ) + np.ones( adjacencyMatrix.shape[0] )
    return labels_pred_spec.astype( int )


def spectralClustering_adjacencyMatrix( matrix, n_clusters ):
    
    #vals, vecs = np.linalg.eigh( matrix )
    #U = vecs[ :, np.argsort( np.abs( vals ) )[-n_clusters:] ]
    
    vals, vecs = np.linalg.eigh( matrix )
    U = vecs[ :, np.argsort( np.abs( vals ) )[ -n_clusters: ] ]
    Lambda = np.diag( vals[ np.argsort( np.abs( vals ) )[ -n_clusters: ] ] )

    
    kmeans = KMeans( n_clusters = n_clusters, n_init = "auto" ).fit( U @ Lambda )
    z = kmeans.labels_

    return z + np.ones( len( z ), dtype = int )



def dealWithZerosInProba( P, epsilon = 0.01 ):
    
    if P[0,0] < 10**(-10):
        P[0,0] = epsilon
        P[0,1] = 1 - epsilon
        
    if P[0,1] < 10**(-10):
        P[0,0] = 1 - epsilon
        P[0,1] = epsilon
        
    if P[1,0] < 10**(-10):
        P[1,0] = epsilon
        P[1,1] = 1 - epsilon
        
    if P[1,1] < 10**(-10):
        P[1,0] = 1 - epsilon
        P[1,1] = epsilon
    
    return P


def asymptoticKullbackLeibler( P, Q ):  
    P = dealWithZerosInProba( P )
    Q = dealWithZerosInProba( Q )

    pi_P = [ 1-P[0,1]/ (1-P[1,1]+P[0,1] ) , P[0,1]/ (1-P[1,1]+P[0,1] ) ]
    
    S = np.zeros( (2) )
    for a in [0,1]:
        S[a] = np.sum ( [ P[a,b] * np.log( P[a,b] / Q[a,b] ) for b in range( 2 ) ] )
    
    return np.sum( [ pi_P[ a ] * S[ a ] ] )


def fromVectorToMembershipMatrice( z, n_clusters = 2 ):
    if len( set ( z ) ) > n_clusters:
        raise TypeError( 'There is a problem with the number of clusters' )
    n = len( z )
    Z = np.zeros( ( n, n_clusters ) )
    for i in range( n ):
        Z[ i, z[i]-1 ] = 1
    return Z


def computeMeans( matrix, labels ):
    N = matrix.shape[0]
    N_in = 0
    N_out = 0
    means_in = 0
    means_out = 0
    
    for i in range( N ):
        for j in range( i+1 ):
            if labels[i] == labels[j]:
                means_in += matrix[ i,j ]
                N_in += 1
            else:
                means_out += matrix[ i,j ]
                N_out += 1
    
    return means_in / N_in, means_out/N_out

def computeMeans_OLD( matrix, labels, n_clusters = 2 ):
    
    Z = fromVectorToMembershipMatrice( labels, n_clusters = n_clusters )
    normalisation = np.linalg.pinv ( Z.T @ Z )
    means = normalisation @ Z.T @ matrix @ Z @ normalisation
    return means[ 0, 0 ], means[ 0, 1 ]


def renyi( a, b, order = 1/2 ):
    return -2 * np.log( np.sqrt( a * b) + np.sqrt( (1-a) * (1-b) ) )



def zeroingDiagonal( M ):
    return M - np.diag( np.diag( M ) )
