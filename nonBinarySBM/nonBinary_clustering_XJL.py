#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:18:53 2023

@author: maximilien
"""

import numpy as np

import nonBinary_baseFunctions as base






def XJL_clustering( X, n_clusters = 2, nBins = 'default', add_noise = False ):
    """
    Clustering Algorithm for non binary networks of the paper 
    [XJL20] Xu, Jog, Loh - 2020 - Optimal rates for community estimation in the weighted stochastic block model
    
    Parameters
    ----------
    X : N-by-N symmetric array of float / int
        Pairwise interactions.

    n_clusters : INT
        Number of clusters.
        
    
        
    nBins: number of bins to use (parameter L in [XJL20])
        the user can provide an int or use 'default'
        the defaut uses the choice in XJL20 of 0.4 * log( log N )^4
        
    add_noise:
        [XJL20] proposed to add noise to avoid issues in the estimation of the pdf.
        But we do not implement it here.
        
    Returns
    -------
    predicted node-labelling z_pred
    """

    N = X.shape[ 0 ]
    
    ### INITIALISATION
    
    if nBins == 'default':
        nBins = int( 0.4 * ( np.log( np.log(N) ) )**4 ) + 1 
    
    A_L = transformationFunction( X, nBins )
    z_pred_each_layers = [ ]
    I = np.zeros( nBins )
    assortativity = [ True for ell in range( nBins ) ]
    for ell in range( nBins ):
        A_ell = np.where( A_L == ell, 1, 0 )
        
        z_ell = base.spectralClustering( A_ell, n_clusters = n_clusters )
        z_pred_each_layers.append( z_ell )
        
        P_ell, Q_ell = base.linkProbabilityEstimation( A_ell, z_ell )
        if P_ell !=0 or Q_ell !=0:
            I[ ell ] = ( P_ell - Q_ell )**2 / ( max( P_ell, Q_ell ) )
        else:
            I[ ell ] = 0
            
        if P_ell < Q_ell:
            assortativity[ ell ] = False

    idealLabelForInitialisation = np.argmax( I )
    z_initialisation = z_pred_each_layers[ idealLabelForInitialisation ]

    
    ### IMPROVEMENT
    z = z_initialisation
    P, Q = [ ], [ ]
    for ell in range( nBins ):
        A_ell = np.where( A_L == ell, 1, 0 )
        """
        if ell <= nBins-2:
            A_ell = np.where( A_L == ell , 1, 0 )
        else:
            A_ell = np.where( A_L >= nBins-1, 1, 0 )
        """
        P_ell, Q_ell = base.linkProbabilityEstimation( A_ell, z_initialisation )
        if P_ell!=0:
            P.append( P_ell )
        else:
            P.append( 2 / N )
            
        if Q_ell != 0:
            Q.append( Q_ell )
        else:
            Q.append( 2 / N )
    
    z = np.zeros( N )
    nodesInEachCommunity = [ ]
    for k in range( n_clusters ):
        nodesInEachCommunity.append ( [ j for j in range( N ) if z_initialisation[j] == k+1 ] )
    
    for i in range( N ):
        li = base.nodeLikelihoods_edgeLabeled_SBM( i, A_L, P, Q, z_initialisation, n_clusters = n_clusters, nodesInEachCommunity = nodesInEachCommunity )
        z[ i ] = np.argmax( li ) + 1
    
    return z.astype( int )



def transformationFunction( X, nBins , dataType = 'real-valued' ):
    Y = np.zeros( X.shape, dtype = int )
    
    if dataType == 'discrete':
        for i in range( X.shape[ 0 ] ):
            for j in range( i ):
                if X[ i, j ] <= nBins - 2:
                    Y[ i, j ] = X[ i, j ]
                else:
                    Y[ i, j ] = nBins - 1
                Y[ j, i ] = Y[ i, j ]
    
    elif dataType == 'real-valued':
        for i in range( X.shape[ 0 ] ):
            for j in range( i ):
                Y[ i, j ] = int( phi( X[ i, j ] ) * nBins )
                if Y[i,j] >= nBins:
                    Y[i,j] = nBins - 1
                    #If X is large, it can happen that phi(X) becomes 1 (with rounding errors) and hence phi(X) nBins = nBins, but we constraint Y to take values in 0,...,nBins-1
                Y[ j, i ] = Y[ i, j ]  
    
    else:
        raise TypeError( 'dataType not implemented' )
    
    return Y
    

def phi( x ):
    if x <0:
        return 1/2 * np.exp( x/2)
    else:
        return 1 - 1/2 * np.exp( -x/2 )


def phi_inverse( x ):
    if x<0 or x > 1:
        raise TypeError( 'x outside range of definition of phi-inverse' )
    elif x == 0:
        return - np.inf 
    elif x ==1:
        return np.inf
    elif x < 1/2:
        return 2 * np.log( 2*x )
    else:
        return -2 * np.log( 2*(1-x) )

