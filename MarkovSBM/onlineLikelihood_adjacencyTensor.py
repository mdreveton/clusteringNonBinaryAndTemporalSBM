#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:40:06 2022

@author: maximilien
"""

import numpy as np
import random as random
from tqdm import tqdm
import spectralClustering as sc





def onlineLikelihoodClustering_knownParameters( adjacencyTensor, initialDistributionRateMatrix, TransitionRateMatrix, initialisation = "Spectral Clustering", K = 2, useTqdm = False ):
    """
    Online clustering when the parameter are known 
    (Algorithm 2 of the paper)
        Step t=1 : use SpectralClustering or RandomGuessing as initialisation
        Then at step t+1, update the prediction using the maximum likelihood estimator
        
    Input: 
        adjacencyMatrix: N x N x T matrix 
        initialDistributionRateMatrix : K x K matrix ( K = number of communities ), whose elements (k,l) is p_{kl} (proba edge between node in community k and node in community l).
        TransitionRateMatrix : a K x K matrix whose element kl is a 2-by-2 matrix ( it is the Markov Transition matrix P_{kl} )
        
        initialisation: the initialisation method (only SpectralClustering and RandomGuessing are implemented)
        
    Output: labelsPred : a matrix N x T, whose slice labelsPred[:,t] gives the predicted community labelling at time t
    """
    N = adjacencyTensor.shape[ 0 ]
    T = adjacencyTensor.shape[ 2 ]
    labelsPred = np.zeros( [ N, T ], dtype = int )
    
    if initialisation.lower( ) == 'random guessing' or initialisation == 'random' or initialisation == 'random guess':
        labelsPred[ :, 0 ] = np.random.randint( 1, K+1, size  = N ) #initialization at random    
    elif initialisation.lower( ) == 'spectral clustering':
        labelsPred[ :, 0 ] = sc.staticSpectralClustering( adjacencyTensor[ :, :, 0 ], K = K )
    else:
        return print("Propose a correct initialisation procedure")

    likelihood = np.zeros( ( N, 2 ) )
    mu1 = initialDistributionRateMatrix[ 0, 0 ]
    nu1 = initialDistributionRateMatrix[ 0, 1 ]
    P = TransitionRateMatrix[ 0, 0, : ]
    Q = TransitionRateMatrix[ 0, 1, : ]
    
    l = np.zeros((2,2))
    l[0,0] = np.log( P[0,0] / Q[0,0] )
    if( P[0,1] * Q[0,1] !=0 ):
        l[0,1] = np.log( P[0,1] / Q[0,1] )
    if(P[1,0] * Q[1,0] !=0):
        l[1,0] = np.log( P[1,0] / Q[1,0] )
    if (P[1,1] * Q[1,1] != 0):
        l[1,1] = np.log( P[1,1] / Q[1,1] )
    
    if P[0,1] == 0:
        l[0,1] = - 9999
    if Q[0,1] == 0:
        l[0,1] = + 9999
    if ( P [0,1] == 0 and Q[0,1] == 0):
        l[0,1] = 0
    if P[1,0] == 0:
        l[1,0] = - 9999
    if Q[1,0] == 0:
        l[1,0] = + 9999
    if ( P[1,0] == 0 and Q[1,0] == 0):
        l[1,0] = 0
    
    ell = np.zeros(2)
    ell[0] = np.log((1-mu1) / (1-nu1))
    ell[1] = np.log (mu1 / nu1)
    
    if useTqdm:
        loop = tqdm(range(1, T) )
    else:
        loop = range(1, T)

    
    M = np.zeros( ( N, N) )
    for i in range( N ):
        for j in range( i-1 ):
            M[ i, j ] = ell[ int(adjacencyTensor[ i, j, 0 ]) ]
            M[ j, i ] = M[ i, j ]
    for t in loop:
        for i in range( N ):
            for j in range( i ):
                M[i,j] += l[ int( adjacencyTensor[i,j,t-1]), int(adjacencyTensor[i,j,t])  ]
                M[j,i] = M[i,j]
                    
        nodesInEachCluster = []
        nodesInEachCluster.append( [ i for i in range( N ) if labelsPred[ i, t-1 ] == 1 ])
        nodesInEachCluster.append( [ i for i in range( N ) if labelsPred[ i, t-1 ] == 2 ])
            
        nodeOrder = [ i for i in range( N ) ]
        random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
        for i in nodeOrder:
            for cluster in range( K ):
                likelihood[ i, cluster ]  = 0
                for node in nodesInEachCluster[ cluster ]:
                    likelihood[ i, cluster ] += M[ i, node ]
            labelsPred[ i, t ] = np.argmax( likelihood[ i, : ] ) + 1
    

    return labelsPred




def onlineLikelihoodClustering_unkownParameters( adjacencyTensor, initialisation = "Spectral Clustering", K = 2, useTqdm = False ):
    """
    Online clustering when the parameters are unknown.
    (Algorithm 3 of the paper)
    
        Step t=1 : use SpectralClustering or RandomGuessing as initialisation
        Then at step t+1, update the prediction using the maximum likelihood estimator

    Input: 
        adjacencyMatrix: N x N x T matrix 
        EdgeRateMatrix : K x K matrix (K = number of communities), whose elements (k,l) is p_{kl} (proba edge between node in community k and node in community l)
        TemporalEdgeEvolutionMatrix  : K x K matrix whose element kl is the Markovian parameter r_{kl}
        
        initialisation: the initialisation method (only SpectralClustering and RandomGuessing are implemented)

    Output: labelsPred : a matrix N x T, whose slice labelsPred[:,t] gives the predicted community labelling at time t
    """
    
    N = adjacencyTensor.shape[0]
    T = adjacencyTensor.shape[2]
    labelsPred = np.zeros( [ N , T ] )
    
    if initialisation.lower( ) == 'random guess' or initialisation == 'random' or initialisation == 'random guessing':
        labelsPred[ :, 0 ] = np.random.randint( 1, K+1, size  = N ) #initialization at random    
    elif initialisation.lower( ) == 'spectral clustering':
        labelsPred[ :, 0 ] = sc.staticSpectralClustering( adjacencyTensor[ :, :, 0 ], K = K )
    else:
        return print("Propose a correct initialisation procedure")
    labelsPred.astype(int)

    likelihood = np.zeros( ( N, K ) )
    transitionsMatrix = np.zeros( ( N, N, 2, 2 )  ) #This will store, at time t, the number of transitions of type (a,b) between nodes i and j (a, b can be 0 or 1)
    
    if useTqdm:
        loop = tqdm(range( 1, T ) )
    else:
        loop = range(1, T)
    
    previous_edge_set = getEdgeSet( adjacencyTensor[ :,:,0 ] )
    edges_set = set( )
    
    for t in loop:
        
        new_edge_set = getEdgeSet( adjacencyTensor[ :,:,t ] )
        for edge in previous_edge_set:
            edges_set.add( edge )
            if edge in new_edge_set or ( edge[1], edge[0] ) in new_edge_set:
                transitionsMatrix[ edge[0], edge[1], 1, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 1 ] += 1
            else:
                transitionsMatrix[ edge[0], edge[1], 1, 0 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 0 ] += 1
        for edge in new_edge_set:
            edges_set.add( edge )
            if ( edge not in previous_edge_set ) and ( ( edge[1], edge[0] ) not in previous_edge_set ) :
                transitionsMatrix[ edge[0], edge[1], 0, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 0, 1 ] += 1

        nodesInEachCluster = [ ]
        for cluster in range( K ):
            nodesInEachCluster.append( [ i for i in range( N ) if labelsPred[i,t-1] == cluster + 1 ] )
        
        #We need estimators for the parameters Pin and Pout (Markov transition probabilities) as well as piin and piout (initial distribution)
        (piin, piout, Pin, Pout) = parameterEstimation( adjacencyTensor[:,:,0], transitionsMatrix, nodesInEachCluster, N , K = K )
        
        #Now that we have estimates for Pin and Pout, we can compute the likelihood
        l = np.zeros( (2,2) )
        
        if Pin[0,0] * Pout[0,0] != 0:
            l[0,0] = np.log( Pin[0,0] / Pout[0,0] )
        else:
            l[0,0] = 0

        if( Pin[0,1] * Pout[0,1] != 0 ):
            l[0,1] = np.log( Pin[0,1] / Pout[0,1] )
        if(Pin[1,0] * Pout[1,0] != 0):
            l[1,0] = np.log( Pin[1,0] / Pout[1,0] )
        if (Pin[1,1] * Pout[1,1] != 0):
            l[1,1] = np.log( Pin[1,1] / Pout[1,1] )
        
        if Pin[0,1] == 0:
            l[0,1] = - 9999
        if Pout[0,1] == 0:
            l[0,1] = + 9999
        if (Pin [0,1] == 0 and Pout[0,1] == 0):
            l[0,1] = 0
        if Pin[1,0] == 0:
            l[1,0] = - 9999
        if Pout[1,0] == 0:
            l[1,0] = + 9999
        if (Pin [1,0] == 0 and Pout[1,0] == 0):
            l[1,0] = 0
        
        
        M = (t-1) * l[0,0] * np.ones( ( N, N) )
        for edge in edges_set:
            i = edge[0]
            j = edge[1]
            M[ i, j ] += l[ 0,1 ] * transitionsMatrix[ i, j, 0, 1 ] 
            M[ i, j ] += l[ 1,0 ] * transitionsMatrix[ i, j, 1, 0 ]
            M[ i, j ] += l[ 1,1 ] * transitionsMatrix[ i, j, 1, 1 ]
            M[ i, j ] += l[0,0] * ( - transitionsMatrix[ i, j, 1, 1 ] - transitionsMatrix[ i, j, 1, 0 ] - transitionsMatrix[ i, j, 0, 1 ] )
            M[ j, i ] = M[ i, j ]

        nodeOrder = [ i for i in range( N ) ] 
        random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
        for i in nodeOrder:
            for cluster in range( K ):
                likelihood[ i, cluster ]  = 0
                for node in nodesInEachCluster[ cluster ]:
                    likelihood[ i, cluster ] += M[ i, node ]
            labelsPred[ i, t ] = np.argmax( likelihood[ i, : ] ) + 1
        
        previous_edge_set = new_edge_set.copy( )


    return (labelsPred, Pin, Pout)



def getEdgeSet( adjacencyMatrix ):
    nonzeros = np.nonzero( adjacencyMatrix ) 
    edge_set = set( )
    for i in range( len( nonzeros[0]) ):
        edge_set.add( ( min( nonzeros[0][i], nonzeros[1][i]), max( nonzeros[0][i], nonzeros[1][i])   ) )

    return edge_set

def parameterEstimation( initial_adjacency_matrix, transitionsMatrix, nodesInEachCluster, n , K = 2 ):
    """
    Return estimators for the parameters Pin and Pout (Markov transition probabilities) as well as piin and piout (initial distribution) 
    given a clustering (nodesInEachCluster) (this can be an estimated clustering)
    """
    Pin = np.zeros( (2,2) )
    Pout = np.zeros( (2,2) )
    piin = 0
    piout = 0
    count = 0
    n0 = 0
    n1 = 0
    for cluster in range(K):
        for i in nodesInEachCluster[cluster]:
            for j in nodesInEachCluster[cluster]:
                if (j!=i):
                    n1 += transitionsMatrix[ i, j, 1, 0 ] + transitionsMatrix[ i, j, 1, 1 ]
                    n0 += transitionsMatrix[ i, j, 0, 0 ] + transitionsMatrix[ i, j, 0, 1 ]
                    
                    Pin[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                    Pin[ 0,0 ] += transitionsMatrix[ i, j, 0, 0] 
                    Pin[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                    Pin[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                    
                    piin += initial_adjacency_matrix[i , j]
                    count += 1
    #print(Pin)
    #print( count, n0, n1 )
    if(count != 0):
        piin = piin / count
    if( n0!=0 ):
        #Pin[1,0] /= n0
        Pin[0,1] /= n0
        Pin[0,0] /= n0
    if (n1 != 0 ):
        #Pin[0,1] /= n1
        Pin[1,0] /= n1
        Pin[1,1] /= n1
    
    count = 0
    n0 = 0
    n1 = 0
    for cluster in range(K):
        for i in nodesInEachCluster[cluster]:
            otherClustersNodes = [dummy for dummy in range(n) if dummy not in nodesInEachCluster[cluster] ]
            for j in otherClustersNodes:
                n1 +=  transitionsMatrix[ i, j, 1, 0 ] + transitionsMatrix[ i, j, 1, 1 ]
                n0 += transitionsMatrix[ i, j, 0, 0 ] + transitionsMatrix[ i, j, 0, 1 ]
                
                Pout[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                Pout[ 0,0 ] += transitionsMatrix[ i, j, 0, 0] 
                
                Pout[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                Pout[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                
                piout += initial_adjacency_matrix[ i, j]
                count += 1
    #print(Pout)
    if (count != 0):
        piout = piout / count
    if( n0!=0 ):
        Pout[0,1] /= n0
        Pout[0,0] /= n0
    if (n1 != 0 ):
        Pout[1,0] /= n1
        Pout[1,1] /= n1
    #print( count, n0, n1)
    
    return (piin, piout, Pin, Pout)
