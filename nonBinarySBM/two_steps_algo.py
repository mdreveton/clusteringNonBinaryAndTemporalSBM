#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 16:21:30 2021

@author: mdreveto
"""

from sklearn.cluster import SpectralClustering, KMeans
import sklearn.cluster as cluster
import numpy as np
import scipy as sp
from tqdm import tqdm 
from sklearn.metrics import accuracy_score



def make_binary_matrix( X, zeroElements = [ 0 ],  dataType = 'discrete' ):
    X_binary = X.copy( )
    
    if dataType == 'discrete':
        """
        for element in zeroElements:
            X_binary = np.where( X_binary == element, 0, X_binary )
        X_binary = np.where( X_binary != 0, 1, 0 )
        """
        X_binary = np.where( X > zeroElements[-1] , 1, 0 )

    elif dataType == 'real valued':
        X_binary = np.where( X > zeroElements[1] , 1, 0 ) + np.where( X < zeroElements[0], 1, 0)
        
    return X_binary

def remove_row( X, row ):
    rows = list( range( X.shape[0] ) )
    rows.remove( row )
    return X[ rows, : ]

def remove_column( X, column ):
    columns = list( range( X.shape[1] ) )
    columns.remove( column )
    return X[ :, columns ]


def spectralClustering( adjacencyMatrix, K = 2, method = "sklearn", assign_labels = 'discretize' ):
    if method == 'sklearn':
        sc = SpectralClustering( n_clusters = K, affinity = 'precomputed', assign_labels = assign_labels )
        labels_pred = sc.fit_predict( adjacencyMatrix ) + np.ones( adjacencyMatrix.shape[ 0 ] )
    
    return labels_pred.astype( int )


def likelihood_test( node, labels_pred, loglikelihoods, K = 2 ):
    n = len( labels_pred )
    loglikelihoods_of_given_node = np.zeros( K )
    for i in range( n ):
        loglikelihoods_of_given_node[ labels_pred[i]-1 ] += loglikelihoods[ node, i ]
                
    return np.argmax( loglikelihoods_of_given_node ) + 1


def consensus( node, labeling1, labeling2, K ):
    """
    here node plays the role of i, 
    labeling1 is \hat{sigma}_i
    and labeling2 is \hat{\sigma}_1 (the baseline reference labeling from which the consensus is considered)
    """
    if len(labeling1) != len(labeling2):
        raise TypeError( 'The two labelings should have the same length' )
    n = len( labeling1 )
    node_in_same_community_as_reference_node_from_labeling1 = [ j for j in range( n ) if labeling1[ j ] == labeling1[ node ] ]
    overlaps = np.zeros( K )
    for k in range( K ):
        node_in_k_from_labeling2 = [ j for j in range( n ) if labeling2[ j ] == k + 1 ]
        overlaps[ k ] = len( set.intersection(set(node_in_same_community_as_reference_node_from_labeling1), set(node_in_k_from_labeling2)) )
    return np.argmax( overlaps ) + 1


def initialisationStep( A, f, g, K = 2, initialisationMethod = 'binary', 
                       zeroElements = [0], nBins = 4 ):
    
    N = A.shape[0]
    
    if initialisationMethod == 'binary' or initialisationMethod == 'Algorithm 1':
    
        A_forInitialisation = make_binary_matrix( A, zeroElements = zeroElements, dataType = f.dataType() )
        
        if f.dataType() == 'discrete':
            if np.sum( [ f.mass_function(x) for x in zeroElements ] ) < np.sum( [ g.mass_function(x) for x in zeroElements ] ):
                associativity = True
            else:
                associativity = False
                A_forInitialisation = np.ones( A.shape ) - A_forInitialisation
                #print( 'Binarisation leads to a non-associative network' )
        
        elif f.dataType() == 'real valued':
            left = zeroElements[ 0 ]
            right = zeroElements[ 1 ]
            p0 = sp.integrate.quad( lambda x : f.mass_function(x) , left, right   ) [0] #intra-cluster proba of 0 in the binary graph
            q0 = sp.integrate.quad( lambda x : g.mass_function(x) , left, right ) [0] #inter-cluster proba of 0 in the binary graph
            if 0 < right and 0 > left:
                p0 += f.mass_function( 0 )
                q0 += g.mass_function( 0 )
            if p0 < q0:
                associativity = True
            else:
                associativity = False
                A_forInitialisation = np.ones( A.shape ) - A_forInitialisation
                #print( 'Binarisation leads to a non-associative network' )
        
        else:
            raise TypeError( 'DataType not implemented for binarisation' )

    
    elif initialisationMethod == 'original XJL20':
        '''
        This corresponds to the original XJL algorithm, 
        in which we select the layer that maximizes an empirical divergence
        No prior knowledge of the interaction distributions is required.
        '''
        
        Y = transformationFunction( A, nBins, dataType = f.dataType( ) )
        I = np.zeros( nBins )
        associativity = [ True for ell in range( nBins ) ]

        for ell in range( nBins ):
            if ell <= nBins-2:
                Y_ell = np.where( Y == ell , 1, 0 )
            else:
                Y_ell = np.where( Y >= nBins-1, 1, 0 )

            labels_pred = spectralClustering( Y_ell, K = K, method = 'sklearn' )
            Pell, Qell, nsame, ndiff = 0, 0, 0, 0
            for i in range( N ):
                for j in range( i ):
                    if labels_pred[ i ] == labels_pred[ j ]:
                        Pell += ( Y_ell[ i, j ] != 0) *1
                        nsame += 1
                    else:
                        Qell += ( Y_ell[ i, j ] != 0) *1
                        ndiff += 1
            if nsame !=0:
                Pell /= nsame
            else:
                Pell = 0
            if ndiff != 0:
                Qell /= ndiff
            else:
                Qell = 0
                
            if Pell !=0 or Qell !=0:
                I[ ell ] = ( Pell - Qell )**2 / ( max( Pell, Qell ) )
            else:
                I[ ell ] = 0
            
            if Pell < Qell:
                associativity[ ell ] = False

        idealLabelForInitialisation = np.argmax( I )
        #print( 'ideal label using best label XJL : ', idealLabelForInitialisation )
        
        A_forInitialisation = np.where( Y == idealLabelForInitialisation , 1, 0 )
        associativity = associativity[ idealLabelForInitialisation ]
        
        if not associativity:
            A_forInitialisation = np.ones( A.shape ) - A_forInitialisation
            print( 'The best label for XJL is not associative' )
        #return spectralClustering( A_forInitialisation, K=K, method='sklearn' )
    
    elif initialisationMethod == 'best label' or initialisationMethod == 'XJL20':
        '''
        This initialisation corresponds to a modified XJL, where we chose the best label using the probability distributions f and g
        '''
        
        I, associativity = getLabelsInformation( f, g, nBins )
        idealLabelForInitialisation = np.argmax( I )
        #print( 'ideal label using best label : ', idealLabelForInitialisation )

        Y = transformationFunction( A, nBins, dataType = f.dataType( ) )
        A_forInitialisation = np.where( Y == idealLabelForInitialisation , 1, 0 )
        associativity = associativity[ idealLabelForInitialisation ]
        #print( 'The ideal label associativity is : ', associativity )
        
        if not associativity:
            A_forInitialisation = np.ones( A.shape ) - A_forInitialisation        
        
    else:
        raise TypeError( 'Initialisation method is not implemented' )
    
    return A_forInitialisation, associativity


def transformationFunction( X, nBins , dataType = 'discrete' ):
    
    Y = np.zeros( X.shape )
    
    if dataType == 'discrete':
        for i in range( X.shape[ 0 ] ):
            for j in range( i ):
                if X[ i, j ] <= nBins - 2:
                    Y[ i, j ] = X[ i, j ]
                else:
                    Y[ i, j ] = nBins - 1
                Y[ j, i ] = Y[ i, j ]
    
    elif dataType == 'real valued':
        for i in range( X.shape[ 0 ] ):
            for j in range( i ):
                Y[ i, j ] = int( phi( X[ i, j ] ) * nBins )
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


def getLabelsProbabilities ( f, g, nBins ):
    P_label, Q_label = np.zeros( nBins ), np.zeros( nBins )
    dataType = f.dataType( )
    
    if dataType == 'discrete':
        for label in range( nBins-1 ):
            P_label[ label ] = f.mass_function( label )
            Q_label[ label ] = g.mass_function( label )
        P_label[ nBins - 1] = 1 - np.sum( P_label )
        Q_label[ nBins - 1] = 1 - np.sum( Q_label )
    
    elif dataType == 'real valued':
        for label in range( nBins ):
            left = phi_inverse( label / nBins )
            right = phi_inverse( ( label+1 ) / nBins )
            
            P_label[ label ] = sp.integrate.quad( lambda x : f.mass_function( x ) , left, right   ) [0]
            Q_label[ label ] = sp.integrate.quad( lambda x : g.mass_function( x ) , left, right ) [0]
            if 0 > left and 0 < right:
                P_label[ label ] += f.mass_function( 0 )
                Q_label[ label ] += g.mass_function( 0 )
    
    else:
        raise TypeError( 'dataType not implemented' )

    return P_label, Q_label


def getLabelsInformation( f, g, nBins ):
    
    P_label, Q_label = getLabelsProbabilities( f, g, nBins )
    I = np.zeros( nBins )
    associativity = np.array( [True for ell in range(nBins) ] )
    for label in range( nBins ):
        I[ label ] = ( P_label[ label ] - Q_label[ label ] )**2 / ( max( 1/1000**3 + P_label[ label ], Q_label[ label ] ) )
        if P_label[ label ] < Q_label[ label ]:
            associativity[ label ] = False
            
    return I, associativity




def two_step_clustering_general_interactions( A, f, g, K = 2, initialisationMethod = 'binary', 
                                             nBins = 5, zeroElements = [0], 
                                             tqdm_ = False, simplified_algo = True ):
    
    N = A.shape[ 0 ]
    A_forInitialisation, associativity = initialisationStep( A, f, g, K = K, initialisationMethod = initialisationMethod, nBins = nBins, zeroElements = zeroElements )
        
    sigma_hat = np.zeros( ( N,N ) , dtype = int )
    sigma = np.zeros( N, dtype = int )
    
    if tqdm_:
        loop = tqdm( range( N ) )
    else:
        loop = range( N )
    
    if simplified_algo :
        
        sigma_tilde = spectralClustering( A_forInitialisation, K = K, method = 'sklearn' )        
        
        sigma = sigma_tilde.copy( )
        
        loglikelihood = lambda x : np.log( ( 1 / N**2 + f.mass_function( x ) ) / ( 1 / N**2 + g.mass_function( x ) ) )
        loglikelihoods = loglikelihood( A )
        
        for i in loop:
            likelihood_prediction_for_node_i = likelihood_test( i, sigma_tilde, loglikelihoods, K = K )
            sigma[ i ] = likelihood_prediction_for_node_i
        
    else:
        for i in loop:
            Atilde = remove_row( A_forInitialisation, i )
            Atilde = remove_column( Atilde, i )
            sigma_tilde = spectralClustering( Atilde, K = K)
            sigma_tilde = list( sigma_tilde )
            sigma_tilde.insert( i, 0 )
            
            #likelihood_prediction_for_node_i = likelihood_test_old( i, sigma_tilde, A, f, g, K = K )
            likelihood_prediction_for_node_i = likelihood_test( i, sigma_tilde, loglikelihoods, K = K )
            
            sigma_tilde[ i ] = likelihood_prediction_for_node_i
            sigma_hat[ :, i ] = sigma_tilde
        
        sigma[ 0 ] = sigma_hat[ 0, 0 ]

        for i in range( 1, N ):
            sigma[ i ] = consensus( i, sigma_hat[ :, i ], sigma_hat[ :, 0 ], K = K )
    
    return sigma

    
def bernoulli( p ):
    return lambda x : x*p + (1-x) * (1-p) if x in [0,1] else TypeError('x should be 0 or 1 to consider Bernoulli' ) 
