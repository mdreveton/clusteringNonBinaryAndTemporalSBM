#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 14:29:23 2023

@author: maximilien
"""

import numpy as np
import temporalSBM_generator as generator
import temporalClustering as tc
import baseFunctions as base
import dataprocessingSocioPattern as dataProcessing


import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import accuracy_score, adjusted_rand_score, adjusted_mutual_info_score



SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18

"""

labels_true = [ ]
for community in range( n_clusters ):
    labels_true = labels_true + [ community+1 for i in range( sizes[community] ) ]
labels_true = np.array( labels_true, dtype = int )


N = 200
T = 100
labels_true = [ 1 for i in range( N//2 ) ] + [ 2 for i in range( N//2 ) ]
block_sizes = [ N//2, N//2 ]
algos = [ 'Algorithm 2', 'time-aggregation', 'ta-laplacian' ]

mu = [ 0.8, 0.2 ]
R = np.array( [ [0.8,0.2], [0.5, 0.5] ] )
Delta_Q = np.array( [ [ 1, - 1 ], [ -1, 1 ] ] )
interactionProbas = [ 1, 1 ]

P = np.array( [ [0.5, 0.5], [0.5, 0.5] ] )
Q = np.array( [ [ 1 - 1/(1+2**(1/2)) ,1/(1+2**(1/2)) ], [ 2 / 1/(1+2**(1/2)), 1 - 2 / 1/(1+2**(1/2))] ] )

initialDistributions = { 'intra' : mu, 'inter' : mu }
transitionMatrices = { 'intra' : P, 'inter' : Q }
adjacencyTensor = generator.generate_markovSBM( T, initialDistributions, transitionMatrices, labels_true, tqdm_= False )

z_2 = clustering( 'Algorithm 2', adjacencyTensor, n_clusters = 2 , step = T )     
z_ta = clustering( 'time-aggregation', adjacencyTensor, n_clusters = 2 , step = T )      
 
max( accuracy_score( labels_true, z_2[:,-1] ), 1 - accuracy_score( labels_true, z_2[:,-1] ) )
max( accuracy_score( labels_true, z_ta[:,-1] ), 1 - accuracy_score( labels_true, z_ta[:,-1] ) )


for algo in algos:
    labels_pred =       
    accuracies[ algo ].append(  )



# =============================================================================
# RUN THIS CODE FOR SYNTHETIC DATA SETS (FIGURE 3)
# =============================================================================

N = 100
T = 200
labels_true = [ 1 for i in range( N//2 ) ] + [ 2 for i in range( N//2 ) ]
block_sizes = [ N//2, N//2 ]
nAverage = 10
epsilon_range = np.linspace( -0.03, 0.03, num = 21 )

algos = [ 'Algorithm 2', 'ta-adjacency', 'ta-laplacian' ]

mu = [ 0.5, 0.5 ]
nu = [ 0.5, 0.5 ]

R = np.array( [ [0.8,0.2], [0.5, 0.5] ] ) #Figure 3a
R = np.array( [ [0.7,0.3], [0.3, 0.7] ] )
R = np.array( [ [0.5,0.5], [0.5, 0.5] ] ) #Figure 3b

Delta_Q = np.array( [ [ 1, - 1 ], [ -1, 1 ] ] )
interactionProbas = [ 1, 1 ]

markov_intra_distributions, markov_inter_distributions = [ ], [ ]
for epsilon in epsilon_range:
    markov_intra_distributions.append( [ mu, R ] )
    markov_inter_distributions.append( [ nu, R + epsilon * Delta_Q ] )

filename = 'markovSBM_varying_epsilon_N_' + str(N) + '_T_' + str(T) + '_R_' + str(R) + '_Delta_' + str(Delta_Q) + '.pdf'
x = epsilon_range
xticks = [ -0.03, -0.015, 0, 0.015, 0.03 ]
xlabel = r'$\epsilon$'


accuracy_mean, accuracy_ste = accuracy_varying_parameter( block_sizes, T, interactionProbas, markov_intra_distributions, markov_inter_distributions, nAverage = nAverage, methods = algos )

plotting( x, accuracy_mean, xticks, 
             curves_errors = accuracy_ste,
             legendTitle = '', figTitle = '',
             xlabel = xlabel, ylabel = 'Accuracy',
             saveFig = False, fileName = filename )



# =============================================================================
# RUN THIS CODE FOR REAL DATA SETS (FIGURE 4)
# =============================================================================

dataset = 'high school 2011'
#The figures of the paper feature: 'high school 2011', 'high school 2012', 'high school 2013'
# But other possible choices: 'workplace 2013', 'workplace 2015' and 'primary school'


if dataset == 'high school 2011':
    communities =  [ ['PC'], ['PC*'], ['PSI*'] ]
    xticks = [ 0, 1000, 2000, 3000, 4000, 5000 ]
elif dataset = 'high school 2012':
    communities =  [ ['PC'], ['PC*'], ['MP*1'], ['MP*2'], ['PSI*'] ]
    xticks = [ 0, 2000, 4000, 6000, 8000, 10000 ]
    xticks = [ 0, 3000, 6000, 9000, 12000 ]
elif dataset = 'high school 2013':
    communities = [ ['2BIO1'], ['2BIO2'], ['2BIO3'], ['MP*1'], ['MP*2'], ['MP'], ['PC'], ['PC*'], ['PSI*'] ]
    xticks = [0, 1500, 3000, 4500, 6000]
    xticks = [0, 2500, 5000, 7500 ]
elif dataset = 'workplace 2013':
    communities = [ ['DISQ'], ['DSE'], ['DMCT'], ['SRH'] ]
    xticks = [0, 1500, 3000, 4500, 6000]
elif dataset = 'workplace 2015':
    communities = [ ['DISQ'],['DMCT'],['DMI'],['DSE'],['DST'] ]
    xticks = [0, 4000, 8000, 12000, 16000]
elif dataset = 'primary school':
    communities = [ ['1A'], ['1B'], ['2A'], ['2B'], 
               ['3A'], ['3B'], ['4A'], ['4B'], 
               ['5A'], ['5B'] ]
    xticks = [0, 1000, 2000, 3000 ]
else:
    raise TypeError( 'The dataset is not implemented' )



temporal_edges_days, labels_true, labels = dataProcessing.dataProcessingSocioPatterns( dataset = dataset, 
                                                                                      communities = communities,
                                                                                      keepNonInteractingPeriods = False )

t = 0
temporal_edges = dict( )
for day in temporal_edges_days.keys( ):
    for u in temporal_edges_days[ day ]:
        temporal_edges[ t ] = temporal_edges_days[ day ][ u ]
        t += 1
      
        
K = len( set( labels_true ) )
if K != len( communities):
    print( 'There is a problem in the number of communities' )
N = len(labels_true)
T = len( temporal_edges )

adjacencyTensor = np.zeros( ( N, N, T ), dtype = int )
for t in range( T ):
    for edge in temporal_edges[t]:
        adjacencyTensor[ edge[0], edge[1], t ] = 1
        adjacencyTensor[ edge[1], edge[0], t ] = 1

step = 20



algos = [ 'Algorithm 2', 'ta-adjacency', 'ta-laplacian' ]

predicted_labels = dict( )
aris = dict( )
accuracies = dict( )
for algo in algos:
    print( algo )
    predicted_labels[ algo ] = clustering( algo, adjacencyTensor, n_clusters = K, step = step, tqdm_ = True )
    aris[ algo ] = [ ]
    accuracies[ algo ] = [ ]
    for t in tqdm( range( predicted_labels[algo].shape[1] ) ):
        aris[algo].append( adjusted_rand_score( labels_true, predicted_labels[algo][ :,t ] ) )
        accuracies[algo].append( computeAccuracy( labels_true, predicted_labels[algo][ :,t ] ) )


x = [ t for t in range( step, T, step ) ]

yticks = np.arange( 0.2, 1.01, step = 0.2 )

plotting( x, accuracies, xticks = xticks,
         yticks = yticks,
             curves_errors = None,
             legendTitle = '', figTitle = '',
             xlabel = 'T', ylabel = 'Accuracy',
             saveFig = False, fileName = 'sociopatterns_' + dataset.replace(" ", "") + '.pdf' )


"""



def clustering( algo, adjacencyTensor, n_clusters, step = 10, tqdm_ = False, spherical = False ):
    
    if algo == 'concatenation' or algo == 'Algorithm 2':
        return tc.spectralConcatenation( adjacencyTensor, n_clusters = n_clusters, step = step, tqdm_ = tqdm_, spherical = spherical )
    
    elif algo == 'time-aggregation-laplacian' or algo == 'ta-laplacian':
        #return tc.spectralTimeAggregated( adjacencyTensor, n_clusters = n_clusters, step = step ) 
        return tc.timeAggregation( adjacencyTensor, n_clusters = n_clusters, step = step, matrix = 'laplacian', tqdm_ = tqdm_ )
    
    elif algo == 'time-aggregation-adjacency' or algo == 'ta-adjacency':
        return tc.timeAggregation( adjacencyTensor, n_clusters = n_clusters, step = step, matrix = 'adjacency', tqdm_ = tqdm_ )
    
    elif algo == 'iterative-SC' or algo == 'it-SC':
        return tc.iterativeSpectral( adjacencyTensor, n_clusters = n_clusters, step = step, n_iter = 10 )
        
    elif algo == 'time-aggregation' or algo == 'time aggregation' or algo == 'ta':
        return tc.spectralTimeAggregated( adjacencyTensor, n_clusters = n_clusters, step = step, tqdm_ = tqdm_, spherical = spherical ) 
        #return tc.timeAggregation( adjacencyTensor, n_clusters = n_clusters, step = step, matrix = 'laplacian' )        
    
    else:
        raise TypeError( 'The algo is not implemented' )


def plotting( x, curves, xticks = None, yticks = [], 
             curves_errors = None,
             legendTitle = '', figTitle = '',
             xlabel = 'a', ylabel = 'ARI',
             saveFig = False, fileName = 'fig.eps'):
        
    if curves_errors is None:
        for key, value in curves.items():
            plt.plot( x, value, label = key )
    else:
        for key, value in curves.items():
            plt.errorbar( x, curves[ key ], yerr = curves_errors[ key ], linestyle = '-.', label = key )

    legend = plt.legend( title = legendTitle, loc='best',  fancybox=True, fontsize= SIZE_LEGEND )
    plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )
    plt.xlabel( xlabel, fontsize = SIZE_LABELS )
    plt.ylabel( ylabel, fontsize = SIZE_LABELS )
    
    if xticks == None:
        plt.xticks( fontsize = SIZE_TICKS )
    else:
        plt.xticks( xticks, fontsize = SIZE_TICKS )
        
    if len( yticks ) == 0:
        plt.yticks( fontsize = SIZE_TICKS )
    else:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
        
    plt.title( figTitle, fontsize = SIZE_TITLE )
    if saveFig:
        plt.savefig( fileName, bbox_inches = 'tight' )
    plt.show()




def severalRuns( block_sizes, T, interactionProbas, markov_intra, markov_inter, algos, nAverage = 10 ):
    
    n_clusters = len( block_sizes )
    
    labels_true = [ ]
    for k in range( n_clusters ):
        labels_true += [ k+1 for i in range( block_sizes[ k ] ) ]

    accuracies = dict( )
    for algo in algos:
        accuracies[ algo ] = [ ]
        
    for run in range( nAverage ):
        p = interactionProbas[0]
        q = interactionProbas[1]

        initialDistributions = { 'intra' : markov_intra[0], 'inter' : markov_inter[0] }
        transitionMatrices = { 'intra' : markov_intra[1], 'inter' : markov_inter[1] }

        if p==1 and q==1:
            adjacencyTensor = generator.generate_markovSBM( T, initialDistributions, transitionMatrices, labels_true, tqdm_= False )
        for algo in algos:
            labels_pred = clustering( algo, adjacencyTensor, n_clusters = n_clusters , step = T )            
            accuracies[ algo ].append( max( accuracy_score( labels_true, labels_pred[:,-1] ), 1 - accuracy_score( labels_true, labels_pred[:,-1] ) ) )

    return accuracies


def accuracy_varying_parameter( block_sizes, T, interactionProbas, markov_intra_distributions, markov_inter_distributions, nAverage = 10, 
                               methods = [ 'Algorithm 1', 'time-aggregation' ] ):
    
    accuracy_mean = dict( )
    accuracy_ste = dict( )
    for method in methods:
        accuracy_mean[ method ] = [ ]
        accuracy_ste[ method ] = [ ]
    
    for i in tqdm( range( len( markov_inter_distributions ) ) ):
        accuracies = severalRuns( block_sizes, T, interactionProbas, markov_intra_distributions[i], markov_inter_distributions[i], methods, nAverage = nAverage )
        for method in methods:
            accuracy_mean[ method ].append( np.mean( accuracies[ method ] ) )
            accuracy_ste[ method ].append( np.std( accuracies[ method ] ) / np.sqrt( nAverage ) )
    
    return accuracy_mean, accuracy_ste



from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)

def computeAccuracy( labels_true, labels_pred ):
    #Compute accuracy by finding the best permutation using Hungarian algorithm
    #See also https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/
    #But be careful, a small difference in the format of the result of linear_assignment of scipy and sklearn
    cm = confusion_matrix( labels_true, labels_pred )
    indexes = linear_sum_assignment(_make_cost_m(cm))
    #js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, indexes[1] ]

    return np.trace( cm2 ) / np.sum( cm2 )

