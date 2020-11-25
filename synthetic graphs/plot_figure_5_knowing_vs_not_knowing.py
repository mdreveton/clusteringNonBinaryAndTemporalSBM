#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:38:15 2020

@author: mdreveto
"""


import numpy as np
import random as random
import matplotlib.pyplot as plt
from tqdm import tqdm 

import MarkovSBM as MarkovSBM
import online_likelihood_clustering_known_parameters as online_algo
import online_likelihood_clustering_unknown_parameters as online_algo_unknown

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

"""

# =============================================================================
# To compare between algo knowing parameter and the one not knowing them
# =============================================================================


cin = 4
cout = 2.5
N = 1000
T = 8
muin = 0.05
muout = 0.035
(qin, qout) = ( 0.6, 0.3 ) #Those are the intra and inter link persistence (number between 0 and 1, close to 1 means high link persistence across time, close to zero means spikes)
Pin = MarkovSBM.makeTransitionMatrix( [1-muin, muin], qin )
Pout = MarkovSBM.makeTransitionMatrix( [1-muout, muout], qout )
TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )
initialDistributionRateMatrix = np.array( [ [muin, muout] , [muout, muin] ] )

( accuracy_knowing, accuracy_not_knowing ) = compareKnowingNotKnowingModelParameters( muin, muout, Pin, Pout, N, T, n_average = 10, savefig = False )


"""

def compareKnowingNotKnowingModelParameters (muin, muout, Pin, Pout, n, T, n_average = 10,
                                             savefig = False, filename = 'comparaison_knowning_parameters.eps'):
    nodesLabels = np.zeros(n)
    for i in range(n//2):
        nodesLabels[i]= 1
    nodesLabels = nodesLabels.astype(int)
    
    accuracy_knowing = np.zeros( ( n_average,T ) )
    accuracy_not_knowing = np.zeros( ( n_average,T ) )
    
    TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )
    initialDistributionRateMatrix = np.array( [ [muin, muout] , [muout, muin] ] )

    for i in tqdm( range(n_average) ):
        random.shuffle(nodesLabels) #shuffle just to be sure the ordering doesn't matter
        ##Create the adjacency matrix
        MSSBM_adja = MarkovSBM.makeMDSBMAdjacencyMatrix( n, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels )
        
        labelsPred = online_algo.likelihoodClustering( MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix, initialisation = "SpectralClustering" )
        accuracy_knowing[i,:] = online_algo.followAccuracy( nodesLabels, labelsPred )
        labelsPredUnkownParameters, Pin_predicted, Pout_predicted = online_algo_unknown.likelihoodClusteringUnkownParameters( MSSBM_adja, initialisation= "SpectralClustering", useTqdm = False)
        accuracy_not_knowing[i,:] = online_algo.followAccuracy( nodesLabels, labelsPredUnkownParameters )
    
    mean_accuracy_knowing = np.zeros( T )
    ste_accuracy_knowing = np.zeros( T )
    mean_accuracy_notknowing = np.zeros( T )
    ste_accuracy_notknowing = np.zeros( T )
    
    for t in range(T):
        mean_accuracy_knowing[t] = np.mean( accuracy_knowing[ :, t ] )
        mean_accuracy_notknowing[t] = np.mean( accuracy_not_knowing[ :, t ] )
        ste_accuracy_knowing[t] = np.std( accuracy_knowing[ :, t ] ) / np.sqrt( n_average )
        ste_accuracy_notknowing[t] = np.std( accuracy_not_knowing[ :, t ] ) / np.sqrt( n_average )
    
    plt.errorbar( list( range(T) ), mean_accuracy_knowing, yerr = ste_accuracy_knowing, linestyle = '--', label='known' )
    plt.errorbar( list( range(T) ), mean_accuracy_notknowing, yerr = ste_accuracy_notknowing, linestyle = '-.', label='unknown' )
    
    legend = plt.legend( title="Model parameters:", loc=4,  fancybox=True, fontsize= SIZE_LEGEND )
    plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )

    plt.xlabel( "Number of time steps", fontsize = SIZE_LABELS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( np.arange( 0,T,2 ), fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    if( savefig ):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    else:
        plt.title( "Evolution of accuracy, \n averaged over %s MSBM with n = %s, T = %s \n and parameters muin = %s, muout = %s, Pin(1,1) = %s and Pout(1,1) = %s" % ( n_average, n, T, muin, muout, Pin[1,1], Pout[1,1] ) )
    plt.show( )
    
    return ( accuracy_knowing, accuracy_not_knowing )

