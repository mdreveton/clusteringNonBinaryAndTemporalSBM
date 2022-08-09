#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 11:44:48 2021

@author: mdreveto
"""
import numpy as np
from tqdm import tqdm
import random as random
import networkx as nx

import referenceAlgorithms as reference


def initialisation_list_edges(N, temporal_edges_for_initialisation , K = 2, initialisation_method = 'union' ):
    
    T = len( temporal_edges_for_initialisation )    
    transitionsMatrix = np.zeros( (N, N, 2, 2) , dtype = int ) #This will store, at time t, the number of transitions of type (a,b) between nodes i and j (a, b can be 0 or 1)
    
    if( initialisation_method == 'union'):
        G = nx.Graph( )
    if ( initialisation_method == 'weighted' or initialisation_method == 'aggregated' or initialisation_method == 'random guess' ):
        G = nx.MultiGraph( )
        
    G.add_nodes_from( [ i for i in range( N ) ] )
    
    for key, value in temporal_edges_for_initialisation.items( ):
        G.add_edges_from( value )
    
    adjacency_matrix = nx.adjacency_matrix( G ).toarray( )
    if ( initialisation_method == 'weighted' or initialisation_method == 'aggregated' or initialisation_method == 'union' ):
        labels_pred = reference.staticSpectralClustering( adjacency_matrix, K = K ) #+ np.ones( N, dtype = 'int8' )
    
    elif initialisation_method == 'random guess':
        labels_pred = np.random.randint( 1, K+1, N )
    
    edges_set = set( )
    for t in range( 1, T ):
        for edge in temporal_edges_for_initialisation[t-1] :
            edges_set.add( edge )
            if edge in temporal_edges_for_initialisation[t] or (edge[1], edge[0]) in temporal_edges_for_initialisation[t]:
                transitionsMatrix[ edge[0], edge[1], 1, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 1 ] += 1
            else:
                transitionsMatrix[ edge[0], edge[1], 1, 0 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 0 ] += 1
        for edge in temporal_edges_for_initialisation[t]:
            if ( edge not in temporal_edges_for_initialisation[t-1] ) or ( ( edge[1], edge[0] ) not in temporal_edges_for_initialisation[t-1] ) :
                transitionsMatrix[ edge[0], edge[1], 0, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 0, 1 ] += 1
    
    nodesInEachCluster = [ ]
    for cluster in range( K ):
        nodesInEachCluster.append( [ i for i in range( N ) if labels_pred[i] == cluster + 1 ] )
    
    ( piin, piout, Pin, Pout ) = parameterEstimation( np.zeros( ( N,N ), dtype = 'int8' ), transitionsMatrix, nodesInEachCluster, N , t = T, K = K )
        
    return labels_pred, Pin, Pout, piin, piout, transitionsMatrix, edges_set




def onlineLikelihoodClustering ( N, temporal_edges, K = 2, timestep = 1, useTqdm = False, initialisation_method = 'random guess' ):
    
    T = len( temporal_edges )
    
    temporal_edges_for_initialisation = dict( )
    for t in range( timestep ):
        temporal_edges_for_initialisation[t] = temporal_edges[ t ]
    
    temporal_edges_for_clustering = dict( )
    for t in range( timestep, T ):
        temporal_edges_for_clustering[ t ] = temporal_edges[ t ]
    
    ( labels_pred_initialisation, Pin, Pout, piin, piout, transitionsMatrix, edges_set ) = initialisation_list_edges( N, temporal_edges_for_initialisation , K = K, initialisation_method = initialisation_method)
    
    labelsPred = np.zeros( [ N, T // timestep + 1 ], dtype = int )
    labelsPred[ :, 0 ] = labels_pred_initialisation
    labelsPred.astype( int )

    likelihood = np.zeros( (  N, K ) )
    #transitionsMatrix = np.zeros( (n, n, 2, 2)  ) #This will store, at time t, the number of transitions of type (a,b) between nodes i and j (a, b can be 0 or 1)
    
    if useTqdm:
        loop = tqdm( range( timestep, T ) )
    else:
        loop = range ( timestep, T )
    
    for t in loop:
        
        for edge in temporal_edges[ t-1 ] :
            edges_set.add( edge ) #This is usefull for later.
            if edge in temporal_edges[t] or ( edge[1], edge[0] ) in temporal_edges[t]:
                transitionsMatrix[ edge[0], edge[1], 1, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 1 ] += 1
            else:
                transitionsMatrix[ edge[0], edge[1], 1, 0 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 0 ] += 1
        
        for edge in temporal_edges[t]:
            if ( edge not in temporal_edges[t-1] ) and ( ( edge[1], edge[0] ) not in temporal_edges[t-1] ) :
                transitionsMatrix[ edge[0], edge[1], 0, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 0, 1 ] += 1
        
        if (t) % timestep == 0:
            
            nodesInEachCluster = [ ]
            for cluster in range( K ):
                nodesInEachCluster.append( [ i for i in range( N ) if labelsPred[ i, t // timestep -1 ] == cluster + 1 ] )
            
            #We need estimators for the parameters Pin and Pout (Markov transition probabilities) as well as piin and piout (initial distribution)
            ( piin, piout, Pin, Pout ) = parameterEstimation( np.zeros( ( N, N ), dtype = 'int8'), transitionsMatrix, nodesInEachCluster, N , t = t, K = K )
            
            #Now that we have estimate of Pin and Pout, we can go ahead and compute the likelihood
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
            
            ell = np.zeros( 2 )
            ell[0] = np.log( (1-piin) / (1-piout) )
            if piin * piout != 0:
                ell[1] = np.log ( piin / piout )
            elif (piin ==0 and piout != 0) or (piin != 0 and piout == 0):
                ell[1] = 9999
            elif piin == 0 and piout == 0 :
                ell[1] = 0
            
            M = (t-1) * l[0,0] * np.ones( ( N, N) )
            for edge in edges_set:
                i = edge[0]
                j = edge[1]
                M[ i, j ] += l[ 0,1 ] * transitionsMatrix[ i, j, 0, 1 ] 
                M[ i, j ] += l[ 1,0 ] * transitionsMatrix[ i, j, 1, 0 ]
                M[ i, j ] += l[ 1,1 ] * transitionsMatrix[ i, j, 1, 1 ]
                M[ i, j ] += l[0,0] * ( - transitionsMatrix[ i, j, 1, 1 ] - transitionsMatrix[ i, j, 1, 0 ] - transitionsMatrix[ i, j, 0, 1 ] )
                M[ j, i ] = M[ i, j ]
            
            likelihood = np.zeros( (  N, K ) )
            nodeOrder = [ i for i in range( N ) ]
            random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
            for i in nodeOrder:
                for cluster in range( K ):
                    for node in nodesInEachCluster[ cluster ]:
                        likelihood[ i, cluster ] += M[ i, node ]
                labelsPred[ i, t // timestep ] = np.argmax( likelihood[ i, : ] ) + 1
            #print( likelihood[38, :] )
            #print( likelihood[1,:] )
            #print( likelihood[10,:] )
        #else:
        #    labelsPred[ :, dummy ] = labelsPred[ :, dummy - 1 ]
    
    #print( edges_set )
    #print( transitionsMatrix[38,50] )
    #print( M[38, 50] )
    #print( transitionsMatrix[0, 1 ] )
    #print( M[0,1] )
    #print( 'likelihood', l )

    return ( labelsPred, Pin, Pout, piin, piout )


def parameterEstimation( initial_adjacency_matrix, transitionsMatrix, nodesInEachCluster, n , t, K = 2):
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
                    n0 += t-1 - ( transitionsMatrix[ i, j, 0, 1] + transitionsMatrix[ i, j, 1, 0]  + transitionsMatrix[ i, j, 1, 1] ) + transitionsMatrix[ i, j, 0, 1 ]

                    Pin[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                    Pin[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                    Pin[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                    
                    piin += initial_adjacency_matrix[i , j]
                    count += 1

    if(count != 0):
        piin = piin / count
    if( n0!=0 ):
        Pin[0,1] /= n0
        Pin[0,0] = 1 - Pin[0,1]
    if (n1 != 0 ):
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
                n0 += t-1 - (transitionsMatrix[ i, j, 0, 1] + transitionsMatrix[ i, j, 1, 0]  + transitionsMatrix[ i, j, 1, 1]) + transitionsMatrix[ i, j, 0, 1 ]
                
                Pout[ 0,1 ] += transitionsMatrix[ i, j, 0, 1] 
                Pout[ 1,0 ] += transitionsMatrix[ i, j, 1, 0] 
                Pout[ 1,1 ] += transitionsMatrix[ i, j, 1, 1] 
                
                piout += initial_adjacency_matrix[ i, j]
                count += 1

    if (count != 0):
        piout = piout / count
    if( n0!=0 ):
        Pout[0,1] /= n0
        Pout[0,0] = 1 - Pout[0,1]
    if (n1 != 0 ):
        Pout[1,0] /= n1
        Pout[1,1] /= n1
    if count == 0:
        piout = 0
        Pout[0,0] = 1
        Pout[1,0] = 1
    
    return (piin, piout, Pin, Pout)



# =============================================================================
# ONLINE ALGORITHM WHEN WE KNOW THE INTERACTION PARAMETER
# =============================================================================

def likelihoodClustering_list_temporal_edge_parameters_known ( N, temporal_edges, Pin, Pout, K = 2, useTqdm = False, timestep_reestimating_clustering = 1, initialisation_method = 'random', initialisation = 'random' ):
    
    T = len( temporal_edges )
    labelsPred = np.zeros( [ N, T ], dtype = int )
        
    if (initialisation == 'random'):
        labelsPred[:,0] = np.random.randint( 1, K+1, size  = N ) #initialization at random    

    l = np.zeros( (2,2) )
    l[0,0] = np.log( Pin[0,0] / Pout[0,0] )
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

    likelihood = np.zeros( (  N, K ) )
    transitionsMatrix = np.zeros( ( N, N, 2, 2)  ) #This will store, at time t, the number of transitions of type (a,b) between nodes i and j (a, b can be 0 or 1)
    edges_set = set( )
    
    if useTqdm:
        loop = tqdm( range( 1, T ) )
    else:
        loop = range (1, T )
    
    for t in loop:
        for edge in temporal_edges[t-1] :
            edges_set.add( edge ) #This is usefull for later.
            if edge in temporal_edges[t] or (edge[1], edge[0]) in temporal_edges[t]:
                transitionsMatrix[ edge[0], edge[1], 1, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 1 ] += 1
            else:
                transitionsMatrix[ edge[0], edge[1], 1, 0 ] += 1
                transitionsMatrix[ edge[1], edge[0], 1, 0 ] += 1
        for edge in temporal_edges[t]:
            if ( edge not in temporal_edges[t-1] ) and ( (edge[1], edge[0]) not in temporal_edges[t-1] ) :
                transitionsMatrix[ edge[0], edge[1], 0, 1 ] += 1
                transitionsMatrix[ edge[1], edge[0], 0, 1 ] += 1
        
        if t % timestep_reestimating_clustering == 0:
            
            nodesInEachCluster = []
            for cluster in range( K ):
                nodesInEachCluster.append( [ i for i in range( N ) if labelsPred[ i, t-1 ] == cluster + 1 ] )

            #Now that we have estimate of Pin and Pout, we can go ahead and compute the likelihood
            M = (t-1) * l[0,0] * np.ones( ( N, N) )
            for edge in edges_set:
                i = edge[0]
                j = edge[1]
                M[ i, j ] += l[ 0,1 ] * transitionsMatrix[ i, j, 0, 1 ] 
                M[ i, j ] += l[ 1,0 ] * transitionsMatrix[ i, j, 1, 0 ]
                M[ i, j ] += l[ 1,1 ] * transitionsMatrix[ i, j, 1, 1 ]
                M[ i, j ] += l[0,0] * ( - transitionsMatrix[ i, j, 1, 1 ] - transitionsMatrix[ i, j, 1, 0 ] - transitionsMatrix[ i, j, 0, 1 ] )
                M[ j, i ] = M[ i, j ]
            
            likelihood = np.zeros( (  N, K ) )
            nodeOrder = [ i for i in range( N ) ]
            random.shuffle( nodeOrder ) #Shuffle the ordering of the nodes
            for i in nodeOrder:
                for cluster in range( K ):
                    for node in nodesInEachCluster[ cluster ]:
                        likelihood[ i, cluster ] += M[ i, node ]
                labelsPred[ i, t ] = np.argmax( likelihood[ i, : ] ) + 1
        else:
            labelsPred[ :, t ] = labelsPred[ :, t - 1 ]
    
    return ( labelsPred )