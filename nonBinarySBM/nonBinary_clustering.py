#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 10:27:05 2023

@author: maximilien
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans

import nonBinary_baseFunctions as base
import general_sbm_generator as generator


interaction_distributions_allowed = [ 'gaussian', 'exponential', 'geometric' ]


def nonBinaryClustering( X, phis, n_clusters = 2, interaction_distribution = 'gaussian', improvement = True, version = 'aggregatingSpectralEmbeddings' ):
    """
    Clustering Algorithm of the paper, for non binary networks
    
    Parameters
    ----------
    X : N-by-N symmetric array of float / int
        Pairwise interactions.

    n_clusters : INT
        Number of clusters.
        
    phis : function
        List of functions.
        
    Returns
    -------
    predicted node-labelling z_pred
    """
    
    ###INITIALISATION STEP
    phiX = [ ]
    
    if phis is None:
        phis = [ lambda x : x, lambda x : np.power( x, 2 ) ]
    
    for phi in phis:
        phiX.append( phi( X ) )
    
    if version not in ['aggregatingSpectralEmbeddings', 'svd' ]:
        version == 'aggregatingSpectralEmbeddings'
    
    if version == 'aggregatingSpectralEmbeddings':
        U = None
        for layer in phiX:
            vals, vecs = np.linalg.eigh( layer )
            U_layer = vecs[ :, np.argsort( np.abs( vals ) )[-n_clusters:] ]
            Lambda_layer = np.diag( vals[ np.argsort( np.abs( vals ) )[-n_clusters:] ] )   
            if U is None:
                U = U_layer @ Lambda_layer
            else:
                U = np.concatenate( [ U, U_layer @ Lambda_layer ], axis = 1 )
            
        kmeans = KMeans( n_clusters = n_clusters, n_init = "auto" ).fit( U )
        z_pred = kmeans.labels_ + np.ones( X.shape[ 0 ], dtype = int )
    
    else:
        M = None
        for layer in phiX:
            if M is None:
                M = layer 
            else:
                M = np.concatenate( [ M, layer ], axis = 1 )
        U, s, Vh = np.linalg.svd( M )
        W = U[ :, :n_clusters ] @ np.diag( s[ :n_clusters ] )
        kmeans = KMeans( n_clusters = n_clusters, n_init = "auto" ).fit( W )
        z_pred = kmeans.labels_ + np.ones( X.shape[ 0 ], dtype = int )
    
    
    ### PARAMETER ESTIMATION + LIKELIHOOD IMPROVEMENT
    if improvement:
        z_pred = refinedClustering( X, z_pred, n_clusters = n_clusters, interaction_distribution = interaction_distribution )
    
    return z_pred.astype( int )


def refinedClustering( X, z_initial, n_clusters = 2, interaction_distribution = 'gaussian', zero_inflated = True ):
    
    if interaction_distribution not in interaction_distributions_allowed:
        print( 'The distribution is not allwed, we skip the improvement' )
        return z_initial
        #raise TypeError( 'The distribution is not allwed' )        
    
    if zero_inflated:
        P = estimateLinkProbabilities( X, z_initial, n_clusters = n_clusters )
        P = np.where( P>1, 1, P )
        P = np.where( P < 0, 0, P )
    else:
        P = np.ones( ( n_clusters, n_clusters ) )


    if interaction_distribution == 'gaussian':
        mu = estimateMeans( X, z_initial, n_clusters = n_clusters )
        #print( mu )
        sigma = estimateStandardDeviation( X, z_initial, n_clusters = n_clusters )
        #print( sigma )
        kernel_pred = dict( )
        for a in range( n_clusters ):
            for b in range( n_clusters ):
                kernel_pred[ a, b ] = generator.zeroInflatedNormal( P[a,b], mu[a,b], sigma[a,b] )
    
    
    elif interaction_distribution == 'exponential':
        mu = estimateMeans( X, z_initial, n_clusters = n_clusters )
        kernel_pred = dict( )
        for a in range( n_clusters ):
            for b in range( n_clusters ):
                if mu[a,b] != 0:
                    kernel_pred[ a, b ] = generator.zeroInflatedGamma( P[a,b], 1, 0, mu[a,b] )
                else:
                    print( 'Problem because mu is zero ')
                    kernel_pred[ a, b ] = generator.zeroInflatedGamma( P[a,b], 1, 0, 1000 )
    
    
    elif interaction_distribution == 'geometric':
        mu = estimateMeans( X, z_initial, n_clusters = n_clusters )
        kernel_pred = dict( )
        for a in range( n_clusters ):
            for b in range( n_clusters ):
                if mu[a,b] != 0:
                    kernel_pred[ a, b ] = generator.zeroInflatedGeometric( P[a,b], 1 / mu[a,b] )
                else:
                    print( 'Problem because mu is zero ')
                    kernel_pred[ a, b ] = generator.zeroInflatedGeometric( P[a,b], 1000 )
    
    z = np.zeros( len(z_initial ) , dtype = int )
    """
    #THE FOLLOWING IS AN OLD (as of Nov 6 2024) code that works but is slow
    for i in range( len(z_initial ) ):
        L = individualNodeLikelihood_homogeneous( i, X[i,:], z_initial, kernel_pred , n_clusters = n_clusters )
        z[ i ] = np.argmax( L ) + 1
    """  
    f_pred = kernel_pred[0,0]
    g_pred = kernel_pred[0,1]

    fX = f_pred.mass_function( X )
    gX = g_pred.mass_function( X )
    logLikelihoodMatrix = np.log( fX / gX )
    
    communities_indices = [ [] for k in range( n_clusters ) ]
    for i in range( X.shape[ 0 ] ):
        communities_indices[ z_initial[i]-1 ].append( i )

    for i in range( len(z_initial) ):
        Li = np.zeros( n_clusters )
        for k in range( n_clusters ):
            Li[ k ] = np.sum( logLikelihoodMatrix[ np.ix_( [i], communities_indices[k] ) ] )
            z[ i ] = np.argmax( Li ) + 1

    return z


def individualNodeLikelihood( node, Xi, z_pred, kernel_pred, n_clusters = 2 ):
    L = np.zeros( n_clusters )
    
    for j in range( len(z_pred) ):
        if j!= node:
            distribution = kernel_pred[ z_pred[node]-1, z_pred[j]-1 ]
            L[ z_pred[j] - 1 ] += np.log( distribution.mass_function( Xi[j] ) )
            #L[ z_pred[j] - 1 ] *= distribution.mass_function( Xi[j] )
    
    return L


def individualNodeLikelihood_homogeneous( node, Xi, z_pred, kernel_pred, n_clusters = 2 ):
    L = np.zeros( n_clusters )
    
    f_pred = kernel_pred[0,0]
    g_pred = kernel_pred[0,1]
    for j in range( len(z_pred) ):
        if j!= node:
            L[ z_pred[j] - 1 ] += np.log(  f_pred.mass_function( Xi[j] ) / g_pred.mass_function( Xi[j] ) )
            #L[ z_pred[j] - 1 ] *= distribution.mass_function( Xi[j] )
    return L

def estimateLinkProbabilities( X, z_initial, n_clusters = 2 ):
    A = np.where( X!=0, 1, 0 )
    Z = base.oneHotRepresentation( z_initial, n_clusters = n_clusters )
    
    P = Z.T @ A @ Z
    normalisation = np.linalg.inv( ( Z.T @ Z ) ) 

    return normalisation @ P @ normalisation
    """
    n = len( z_initial )
    number_nodePair_insideSameCommunity = np.trace( (Z.T @ Z) @ (Z.T @ Z - np.eye(n_clusters) ) )
    number_nodePair_inDifferentCommunities = n * (n-1) - number_nodePair_insideSameCommunity
    
    p_pred = np.trace( P ) / number_nodePair_insideSameCommunity #Tr P gives the number of intra-community interactions
    q_pred = ( np.sum( P ) - np.trace( P ) ) / number_nodePair_inDifferentCommunities #sum P - Tr P gives the number of inter-community interactions
    """
    #return p_pred, q_pred
    

def estimateMeans( X, z_initial, n_clusters = 2 ):
    A = np.where( X!=0, 1, 0 )
    Z = base.oneHotRepresentation( z_initial, n_clusters = n_clusters )
    
    mu = Z.T @ X @ Z
    #normalisation = np.linalg.inv( ( Z.T @ A @ Z ) ) 
    #return normalisation @ mu @ normalisation #The code in comment was only true for dense interactions and not zero-inflated ones.
    normalisation = Z.T @ A @ Z
    return np.divide( mu, normalisation )
    

def estimateStandardDeviation( X, z_initial, n_clusters = 2 ):
    A = np.where( X!=0, X**2, 0 )
    Z = base.oneHotRepresentation( z_initial, n_clusters = n_clusters )
    
    variance = Z.T @ A @ Z
    normalisation = np.linalg.inv( ( Z.T @ Z ) ) 
    
    variance = normalisation @ variance @ normalisation - 1/len(z_initial) * np.power( estimateMeans( X, z_initial, n_clusters = n_clusters ) , 2 )
    return np.power( variance, 1/2 )

    