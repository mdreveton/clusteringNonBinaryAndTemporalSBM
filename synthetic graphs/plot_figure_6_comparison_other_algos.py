#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:06:43 2020

@author: mdreveto
"""



import numpy as np
import random as random
import matplotlib.pyplot as plt
from tqdm import tqdm 
from sklearn.metrics import accuracy_score


import MarkovSBM as MarkovSBM
import online_likelihood_clustering_known_parameters as online_algo
import online_likelihood_clustering_unknown_parameters as online_algo_unknown
import baseline_clustering_algorithms as baseline_algo

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18


"""
# =============================================================================
# Figure 6: Comparison of different clustering algorithms
# =============================================================================


n_average = 20
N = 500
T = 30
muin = 0.05
muout = 0.04

algosToCompare = ['online likelihood', 'union graph SC', 'time aggregated SC', 'squared adjacency SC', 'best friends forever', 'enemies of my enemy']


# Figure 6b) : Pin varies, Pout stay (iid evolution for inter-community interaction pattern)

scenarios = []
qinTested = [ 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0 ]
#qinTested = [ 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 1.0 ]
#qinTested = [ 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 1.0 ]
for qin in qinTested:
    scenarios.append( [ muin, muout, qin, muout ] )
mean_accuracy, ste_accuracy = resultsDifferentAlgo( N, T, scenarios, n_average, algosToCompare = algosToCompare )
plotResultsDifferentAlgo(scenarios, mean_accuracy, ste_accuracy, savefig = True, filename = 'comparison_algo_iid_outside_varying_Pin(1,1).eps', whatVaries = 'qin', algosToCompare = algosToCompare, withLegend = False)



# Figure 6a) : Pout varies, Pin(1,1) stay equal to 1 (static evolution for inter-community interaction paterns)

scenarios = []
#qoutTested = [ mu ,  0.1, 0.2, 0.28, 0.3, 0.32, 0.35, 0.4, 0.45, 0.5 ] 
qoutTested = [ muout ,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 ] 
for qout in qoutTested:
    scenarios.append( [ muin, muout, 1, qout] )
mean_accuracy, ste_accuracy = resultsDifferentAlgo( N, T, scenarios, n_average, algosToCompare = algosToCompare )
plotResultsDifferentAlgo(scenarios, mean_accuracy, ste_accuracy, savefig = False, filename = 'comparison_algo_static_inside_varying_Pout(1,1).eps', whatVaries = 'qout', algosToCompare = algosToCompare)


"""



def compareAlgo(nAverage, n, T, muin, muout, qin, qout, algosToCompare = ['likelihood-SC', 'SC-union_graph', 'SC-weighted_union_graph', 'naive', 'ennemy'] ):
    initialDistributionRateMatrix = np.array( [ [muin, muout] , [muout , muin] ] )
    
    pin = initialDistributionRateMatrix [ 0, 0 ]
    pout = initialDistributionRateMatrix [ 0, 1 ]
    Pin = MarkovSBM.makeTransitionMatrix( [ 1-pin, pin ], qin )
    Pout = MarkovSBM.makeTransitionMatrix( [ 1-pout, pout ], qout )
    
    TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )
    
    accuracy = {}
    for algo in algosToCompare:
        accuracy[algo] = []
#    print(computeThreshold(cin, cout, 1-qin,  1-qout, T))
    
    for i in range( nAverage ):
        nodesLabels = np.zeros(n)
        for j in range(n):
            nodesLabels[j]= ( random.random() > 0.5 ) * 1
        nodesLabels = nodesLabels.astype( int )
        MSSBM_adja = MarkovSBM.makeMDSBMAdjacencyMatrix( n, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels )
        
        for algo in algosToCompare:
            if algo == 'likelihood-SC' or algo == 'online likelihood':
                labelsPred = online_algo.likelihoodClustering( MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix,  initialisation = "SpectralClustering" )
                labelsPred = labelsPred[:,-1]
                
            elif algo == 'likelihood-RG':
                labelsPred = online_algo.likelihoodClustering( MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix,  initialisation = "RandomGuessing" )
                labelsPred = labelsPred[:,-1]
                
            elif algo == 'likelihood-SC-estimation' or algo == 'likelihood unknown parameters':
                labelsPred, Pin_predicted, Pout_predicted = online_algo_unknown.likelihoodClusteringUnkownParameters( MSSBM_adja, initialisation = "SpectralClustering" )
                labelsPred = labelsPred[:,-1]
                
            elif algo == 'SC-union_graph' or algo == 'union graph SC':
                unionGraph = baseline_algo.simpleUnionGraph( MSSBM_adja )
                labelsPred = baseline_algo.staticSpectralClustering( unionGraph )
                
            elif algo == 'SC-weighted_union_graph' or algo == 'time aggregated SC':  
                weightedUnionGraph = baseline_algo.weightedUnionGraph( MSSBM_adja )
                labelsPred = baseline_algo.staticSpectralClustering( weightedUnionGraph )
                
            elif algo == 'naive' or algo == 'best friends forever':
                (n_predicted_clusters, labelsPred, isolated_nodes) = baseline_algo.naiveAlgorithm( MSSBM_adja )
                
            elif algo == 'ennemy' or algo == 'enemies of my enemy':
                (n_labels, labelsPred) = baseline_algo.ennemiesOfMyEnnemiesClustering( MSSBM_adja )
                
            elif algo == 'SC_on_squared_adjacency' or algo == 'squared adjacency SC':
                labelsPred = baseline_algo.spectralClusteringOnSquareMatrix(MSSBM_adja)
            
            accuracy[ algo ].append( max (accuracy_score(nodesLabels, labelsPred), 1 - accuracy_score(nodesLabels, labelsPred) ) )
    
    return accuracy


def resultsDifferentAlgo( n, T, scenarios, n_average, algosToCompare = ['likelihood-SC', 'SC-union_graph', 'SC-weighted_union_graph', 'naive', 'ennemy'] ):
    
    mean_accuracy = np.zeros( (len(algosToCompare), len(scenarios ) ) )
    ste_accuracy = np.zeros( (len(algosToCompare), len(scenarios) ) )

    for s in tqdm( range( len( scenarios) ) ):
        print( s )
        
        cin = scenarios[s][0]
        cout = scenarios[s][1]
        qin = scenarios[s][2]
        qout = scenarios[s][3]
        
        accuracy = compareAlgo( n_average, n, T, cin, cout, qin, qout, algosToCompare = algosToCompare )
        
        for algo in range( len(algosToCompare) ):
            mean_accuracy[ algo, s ] = np.mean( accuracy[ algosToCompare[algo] ] )
            ste_accuracy [ algo, s ] = np.std( accuracy[ algosToCompare[algo] ] ) / np.sqrt( n_average )
        
    return mean_accuracy, ste_accuracy
    
        
    
def plotResultsDifferentAlgo( scenarios, mean_accuracy, ste_accuracy , algosToCompare = ['likelihood-SC', 'SC-union_graph', 'SC-weighted_union_graph', 'naive', 'ennemy'],
                                  savefig = False, filename = 'filename', whatVaries = 'qout' , withLegend = True):
    if whatVaries == 'qin':
        qout = scenarios[0][3]
        qin = []
        for i in range(len(scenarios)):
            qin.append( scenarios[i][2] )
        for algo in range( len(algosToCompare) ):
            plt.errorbar( qin, mean_accuracy[algo,:], yerr = ste_accuracy[algo,:], linestyle = '-.', label = str( algosToCompare[algo] ) )
        plt.xlabel("Pin(1,1)", fontsize = SIZE_LABELS)
            
    elif whatVaries == 'qout' :
        qin = scenarios[0][2]
        qout = []
        for i in range(len(scenarios)):
            qout.append( scenarios[i][3] )
        for algo in range( len(algosToCompare) ):
            plt.errorbar( qout, mean_accuracy[algo,:], yerr = ste_accuracy[algo,:], linestyle = '-.', label = str( algosToCompare[algo] ) )
        plt.xlabel("Pout(1,1)", fontsize = SIZE_LABELS)

    if(withLegend): 
        legend = plt.legend( title= "Algorithm:", loc= 6, fancybox= True , fontsize = 12)
        plt.setp(legend.get_title(),fontsize= 12)

    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
#    plt.title("\n Evolution of accuracy, \n averaged over %s MSBM with n = %s, T = %s, \n muin = %s / n, muout = %s /n" % (n_average, n, T, cin, cout))
    plt.xticks( fontsize= SIZE_TICKS )
    plt.yticks( fontsize= SIZE_TICKS )

    if (savefig):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    plt.show()

    return 





