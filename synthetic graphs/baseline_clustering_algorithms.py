#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 14:56:33 2020

@author: mdreveto
"""

import numpy as np
import networkx as nx

import scipy as sp
from sklearn.cluster import SpectralClustering




def staticSpectralClustering(adjacencyMatrix, n_clusters = 2):
    sc = SpectralClustering(n_clusters = n_clusters, affinity='precomputed', assign_labels='discretize')
    labels_pred_spec = sc.fit_predict(adjacencyMatrix)
    return labels_pred_spec




#### Some special graphs

def weightedUnionGraph(MSSBM_adja):
    adjacency_matrix = np.array( MSSBM_adja[:,:,0] )
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(i):
            adjacency_matrix[i,j] = sum(MSSBM_adja[i,j,:])
            adjacency_matrix[j,i] = adjacency_matrix [i,j]
    return adjacency_matrix


def simpleUnionGraph( MSSBM_adja ):
    adjacency_matrix = np.array( MSSBM_adja[:,:,0] )
    n = adjacency_matrix.shape[0]
    for i in range(n):
        for j in range(i):
            if ( sum( MSSBM_adja[i,j,:] ) >= 1 ):
                adjacency_matrix[i,j] = 1
                adjacency_matrix[j,i] = adjacency_matrix [i,j]
    return adjacency_matrix


def intersectionGraph(MSSBM_adja):
    """
    Return the adjacency matrix of the intersection graph
    """
    T = MSSBM_adja.shape[2]
    
    return np.sum(MSSBM_adja, axis = 2) //T


###### Naive algorithms


def naiveAlgorithm(MSSBM_adja):
    """
    Implementaton of the Naive algorithm of the paper
    """    
    adjacency_matrix_intersection_graph = intersectionGraph(MSSBM_adja)    
#    graphOfTconnectedNodes = sp.sparse.csr_matrix(adjacency_matrix_intersection_graph)
#    n_components, labels = sp.sparse.csgraph.connected_components(csgraph = graphOfTconnectedNodes, directed=False, return_labels=True)
    Ginter = nx.from_numpy_matrix( adjacency_matrix_intersection_graph )
    
    isolated_nodes = []
    list_nodes_connected_components = []
    for c in nx.connected_components(Ginter):
        if len(c) == 1:
            isolated_nodes += [elt for elt in c]
        else:
            list_nodes_connected_components.append([elt for elt in c])
    
    n_predicted_clusters = len( list_nodes_connected_components )
    labels_pred = np.zeros( len(MSSBM_adja[:,0,0] ) )    
    for cluster in range( n_predicted_clusters ):
        for node in list_nodes_connected_components[cluster]:
            labels_pred[ node ] = cluster
    for node in isolated_nodes:
        labels_pred[ isolated_nodes ] = np.random.randint(0 , n_predicted_clusters + 1)

    return (n_predicted_clusters, labels_pred, isolated_nodes)





def ennemiesOfMyEnnemiesClustering(MSSBM_adja):
    """
    Cluster a Markov SBM using the algorithm "ennemy of my ennemy"
    Recall two nodes are called ennemies if at (at least) one snapshot, we see an edge appearing or disappearing between them
    """
    T = MSSBM_adja.shape[2] 
    
    ennemyAdjacencyMatrix = np.sum(MSSBM_adja, axis = 2) #the adjacency matrix where element (ij) equal 1 iff nodes i and j are ennemies, equal 0 otherwise
    ennemyAdjacencyMatrix = ( ( (ennemyAdjacencyMatrix ==0)*1 + (ennemyAdjacencyMatrix == T)*1 ) + 1 )%2
    
    return findFriendsAmongEnemies(ennemyAdjacencyMatrix)
    

def findFriendsAmongEnemies(ennemyAdjacencyMatrix):
    """
    For an adjacency matrix giving the ennemies,
    labels the nodes such that: two nodes have the same labels (i.e. they are friend) iff they share a common ennemy
    """
    n = ennemyAdjacencyMatrix.shape[0]
    labels = np.zeros(n)
    
    queue = list(range(n))
    n_labels=0
    while(queue != []):
        currentNode = queue[0]
        labels[currentNode] = n_labels
        nodesExceptCurrentOne = list(range(n))
        nodesExceptCurrentOne.remove(currentNode)
        for i in list(nodesExceptCurrentOne) :
            if(hasEnnemyInCommon(ennemyAdjacencyMatrix, currentNode, i)):
                labels[i] = n_labels
                if(i not in queue):
                    #return print("Error : the adjacency matrix made of ennemy nodes is not bipartite, so the algorithm cannot give a proper solution")
                    return ( 0, np.zeros( n ) )
                else:
                    queue.remove( i )
        n_labels += 1
        queue.remove(currentNode)
        
    return (n_labels, labels)


def hasEnnemyInCommon(ennemyAdjacencyMatrix, i,j ):
    n = ennemyAdjacencyMatrix.shape[0]
    ennemiesOfi = [node for node in range(n) if ennemyAdjacencyMatrix[i,node] == 1 ]
    ennemiesOfj = [node for node in range(n) if ennemyAdjacencyMatrix[j,node] == 1 ]
    return ( not set(ennemiesOfi).isdisjoint(ennemiesOfj) )






###########################
    

def spectralClusteringOnSquareMatrix(adjacencyMatrix , K = 2, biais_adjusted = True):
    """
    Correspond to the algo studied in the paper:
        Jing  Lei. Tail  bounds  for  matrix  quadratic  forms  and bias  adjusted  spectral  clustering  in  multi-layer  stochastic  block  models.
    """
    T = adjacencyMatrix.shape[2]
    n = adjacencyMatrix.shape[0]
    labels_pred = np.zeros(n, dtype = int)
    
    if(biais_adjusted):
        D = sp.sparse.dia_matrix( np.diag( sum( adjacencyMatrix[ :,:,0 ] ) ) )
        adjaSquaredSum = sp.sparse.csr_matrix( adjacencyMatrix[ :,:,0 ]) @  sp.sparse.csr_matrix( adjacencyMatrix[ :,:,0 ] ) - D
    else:
        adjaSquaredSum = sp.sparse.csr_matrix( adjacencyMatrix[ :,:,0 ]) @  sp.sparse.csr_matrix( adjacencyMatrix[ :,:,0 ] )
    
    for t in range(1,T):
        if(biais_adjusted):
            D = sp.sparse.dia_matrix( np.diag( sum( adjacencyMatrix[ :,:,t ] ) ) )
            adjaSquaredSum += sp.sparse.csr_matrix( adjacencyMatrix[ :,:,t ] ) @ sp.sparse.csr_matrix( adjacencyMatrix[ :,:,t ] ) - D
        else:
            adjaSquaredSum += sp.sparse.csr_matrix( adjacencyMatrix[ :,:,t ] ) @ sp.sparse.csr_matrix( adjacencyMatrix[ :,:,t ] )
    
    vals, vecs = sp.sparse.linalg.eigsh( adjaSquaredSum.asfptype() , k=2, which = 'LM')
    secondVector = vecs[:,0]
    for i in range(n):
        labels_pred[i] = ( secondVector[i] > 0 ) *1
    
    return labels_pred