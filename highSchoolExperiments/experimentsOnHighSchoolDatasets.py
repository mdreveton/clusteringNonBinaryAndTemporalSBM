#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 14:21:27 2022

@author: maximilien
"""

import numpy as np
import scipy as sp
import random as random
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from tqdm import tqdm 
from itertools import permutations 
from itertools import combinations


#import os
#working_directory_path = os.getcwd() # Check current directory's path
#os.chdir(working_directory_path)
import preprocessingHighSchool as highSchool
import onlineLikelihood_temporalEdges as onlineLikelihood
import referenceAlgorithms as reference



SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18


"""
----------------------
year = 2011
----------------------
year = 2012
----------------------
year = 2013
----------------------

communities =  [ ['PC'], ['PC*'], ['PSI*'] ]
K = len( communities )
labels_true, temporal_edges, days, node_indexing, sex = highSchool.preprocess_high_school_dataset_temporal_edges( groups_considered = communities, year = year )
N = len(labels_true)
T = len( temporal_edges )

---------------------------

timestep = 20
n_average = 1
assign_labels = 'kmeans'

adjacencyTensor = np.zeros( ( N, N, T ) )
for t in range( T ):
    for edge in temporal_edges[t]:
        adjacencyTensor[ edge[0], edge[1], t ] += 1
        adjacencyTensor[ edge[1], edge[0], t ] += 1

accuracies = dict( )
mean_accuracies = dict( )
ste_accuracies = dict( )

methods = ['Algorithm 3', 'mean-adjacency', 'mean-norm-Lap', 'sum-of-squared' ]

for method in methods:
    accuracies[ method ] = np.zeros( ( n_average, T // timestep + 1 ) )
    mean_accuracies[ method ] = np.zeros( T // timestep + 1 )
    ste_accuracies[ method ] = np.zeros( T // timestep + 1 )

for run in tqdm( range( n_average ) ) :
    for method in methods:
        labelsPred = temporalClustering( method, temporal_edges, adjacencyTensor, K = K, timestep = timestep, assign_labels = assign_labels, useTqdm = False )
        accuracies[ method ][ run, : ] = follow_accuracy_several_clusters( labels_true, labelsPred, K = K, useTqdm = False )


for t in range( T // timestep + 1 ):
    for method in methods:
        mean_accuracies[ method ][ t ] = np.mean( accuracies[ method ][:,t] ) 
        ste_accuracies[ method ][ t ] = np.std( accuracies[ method ][:,t] ) / np.sqrt( n_average )



if year == 2011:
    xticks = np.arange( 0, T//timestep+1, 75 )
elif year == 2012:
    xticks = np.arange( 0, T//timestep+1, 150 )
elif year == 2013:
    xticks = np.arange( 0, T//timestep+1, 100 )


titleFig = ''
filename = 'high_school_' + str(year) + '_timestep_' + str(timestep) + '_nAverage_' + str( n_average ) + '_assignLabels_' + str(assign_labels) + '.eps'
xlabels = xticks * timestep
plot_results( mean_accuracies, methods, xticks = xticks, xlabels = xlabels, std_accuracies = [], titleFig = titleFig, saveFig = False, filename = filename, legend_title = '' )

"""


def temporalClustering( method, temporal_edges, adjacencyTensor, K = 2, timestep = 10, assign_labels = 'kmeans', useTqdm = False ):
    
    if method == 'online-likelihood' or method == 'Algorithm 3':
        N = adjacencyTensor.shape[ 0 ]
        ( labelsPred, Pin, Pout, piin, piout ) = onlineLikelihood.onlineLikelihoodClustering( N, temporal_edges , K = K, timestep = timestep, initialisation_method = 'weighted', useTqdm = useTqdm )
        return labelsPred
    
    elif method == 'mean-adjacency SC' or method == 'mean-adjacency':
        return reference.meanAdjacencyMatrixSpectralClustering( adjacencyTensor, K = K, assign_labels = assign_labels, timestep = timestep, useTqdm = useTqdm )

    elif method =='sum-of-squared SC' or method == 'sum-of-squared':
        return reference.sumOfSquaredSpectralClustering( adjacencyTensor, K = K, biais_adjusted = True, timestep = timestep, assign_labels = assign_labels, useTqdm = useTqdm )
    
    elif method == 'time-aggregated SC' or method == 'mean-norm-Lap':
        return reference.timeAggregatedSpectralClustering( adjacencyTensor, K = K, timestep = timestep, assign_labels = assign_labels, useTqdm = useTqdm )
    
    else:
        raise TypeError( 'Method not implemented' )
    





def plot_results( accuracies, methods, xticks, xlabels, yticks = [ ], std_accuracies = [], titleFig = 'Title', saveFig = False , filename = 'Fig.eps', legend_title = 'Algorithm', legend = True ):
    
    if std_accuracies == []:
        for method in methods:
            plt.plot( accuracies[method], label = method )
    else:
        for method in methods:
            plt.errorbar( range( len(accuracies[method]) ), accuracies[ method ], yerr = std_accuracies[ method ], linestyle = '-.', label= method )
    
    if legend:
        legend = plt.legend( title=legend_title, loc='best',  fancybox=True, fontsize= SIZE_LEGEND )
        plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )

    plt.xlabel( "Number of snapshots", fontsize = SIZE_LABELS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( xticks, labels = xlabels, fontsize = SIZE_TICKS )
    if yticks == [ ] or yticks == None:
        plt.yticks( fontsize = SIZE_TICKS )
    else:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
    if( saveFig ):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    else:
        plt.title( titleFig, fontsize = SIZE_TITLE )
    plt.show( )

    return 0



def accuracy_several_clusters( labels_true, labels_pred, K = 2):
    accuracy = 0
    best_perm = []
    for perm in permutations( [ i + 1 for i in range(K) ] ): #permutations over the set {1,2,\dots, K }
        labels_pred_perm = [ perm[label-1] for label in labels_pred ] 
        if  accuracy_score(labels_true, labels_pred_perm ) > accuracy:
            accuracy = accuracy_score(labels_true, labels_pred_perm )
            best_perm = perm
    return accuracy, best_perm


def follow_accuracy_several_clusters( labels_true, labelsPred, K = 2, useTqdm = True ):
    """
    This function compute the accuracy of the labelling labelsPred at each timestep t
    """
    accuracy = []
    if useTqdm:
        loop = tqdm( range( labelsPred.shape[1] ) )
    else:
        loop = range( labelsPred.shape[1] )
        
    for t in loop:
        acc, perm = accuracy_several_clusters( labels_true, labelsPred[:,t], K = K )
        accuracy.append( acc )
    return accuracy
