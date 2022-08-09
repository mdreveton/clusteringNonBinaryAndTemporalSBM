#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 14:44:03 2022

@author: maximilien
"""

import numpy as np
import random as random
import matplotlib.pyplot as plt
from tqdm import tqdm 

from sklearn.metrics import accuracy_score

import MarkovSBM as MarkovSBM
import onlineLikelihood_adjacencyTensor as onlineLikelihood
import twoStepAlgo_markovSBM as twoStepAlgo


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18


"""
# =============================================================================
# Figure 4: Algorithm 1 on Markov SBM
# =============================================================================

nAverage = 10
N = 400
K = 2
T = int( 10 * np.log(N) ) + 1

a_tested = [3, 9]
p11 = 0.3
q11_range = np.linspace( 0.01, 0.99, num = 25 )

( mean_accuracy, ste_accuracy ) = performanceAlgo1( N, K, T, a_tested, p11, q11_range, nAverage = nAverage )

filename = 'algo1_Markov_N_' + str(N) + '_T_' + str(T) + '_p11_' + str(p11) + '_nAverage_' + str(nAverage) + '.eps'
plotFigure( q11_range, mean_accuracy, ste_accuracy, a_tested, legend_title = 'a', xlabel = "$Q_{11}$", savefig=False, fileName=filename)




# =============================================================================
# Figure 5: Comparison between initialization methods in Algo 2 (random guess vs spectral clustering)
# =============================================================================

n_average = 25
T = 20
N = 400
cin = 15
cout = 5
P11 = 0.7 #qin repesent P(1,1)
Q11 = 0.4 #represents Q(1,1)

mu1, nu1 = cin / N , cout / N 
P = makeTransitionMatrix( [ 1-mu1, mu1 ], P11 )
Q = makeTransitionMatrix( [ 1-nu1, nu1 ], Q11 )

filename = 'comparison_initialization_cin_' + str( cin ) + '_cout_' + str(cout) + '_P11_' + str(P11) + '_Q11_' + str(Q11) + '.eps'
(accuracy_SC, accuracy_RG) = compareInitialization( mu1, nu1, P, Q, N, T, n_average = n_average, savefig = False, filename = filename )





# =============================================================================
# Figure 6: Comparison of Algo 2 and Algo 3 (known parameters VS unknown parameters)
# =============================================================================

n_average = 25
N = 400
T = 50
cin = 8
cout = 2
mu1 = cin / N
nu1 = cout / N
( P11, Q11 ) = ( 0.6, 0.3 ) #Those are the intra and inter link persistence (number between 0 and 1, close to 1 means high link persistence across time, close to zero means spikes)
P = makeTransitionMatrix( [ 1-mu1, mu1 ], P11 )
Q = makeTransitionMatrix( [ 1-mu1, nu1 ], Q11 )

filename = 'comparison_online_knowingNotKnowing_N_' + str( N ) + '_T_' + str( T ) + '_mu1_' + str(mu1) + '_nu1_' + str(nu1) + '_p11_' + str(P11) + '_q11_' + str(Q11) + '_nAverage_' + str(n_average) + '.eps'
( accuracy_knowing, accuracy_not_knowing ) = compareKnowingNotKnowingModelParameters( mu1, nu1, P, Q, N, T, n_average = n_average, savefig = False, filename = filename )



"""

def performanceAlgo1( N, K, T, a, p11, q11_range, nAverage = 10 ):
    
    accuracies = dict( )
    mean_accuracy, ste_accuracy = dict( ), dict( )
    for dummy in a:
        accuracies[ dummy ] = [ ] 
        mean_accuracy[ dummy ], ste_accuracy[ dummy ] = [ ], [ ]
    
    nodesLabels = np.ones( N, dtype = int )
    for i in range( N//2, N ):
        nodesLabels[ i ] = 2

    for i in tqdm( range( len ( q11_range ) ) ):
        q11 = q11_range[ i ]
        for dummy in a:
            mu1, nu1 = dummy / N, dummy / N
            P = makeTransitionMatrix( [ 1-mu1, mu1 ], p11 )
            Q = makeTransitionMatrix( [ 1-nu1, nu1 ], q11 )

            transitionRateMatrix = np.array( [ [ P, Q ], [ Q, P ] ] )
            initialDistributionRateMatrix = np.array( [ [ mu1, nu1 ], [ nu1, mu1 ] ] )

            accuracies[ dummy ] = [ ] 
            for run in range( nAverage ):
                adjacencyTensor = MarkovSBM.makeMDSBMAdjacencyMatrix( N, T, initialDistributionRateMatrix, transitionRateMatrix, nodesLabels )

                labelsPred = twoStepAlgo.twoStepAlgo_MarkovSBM( adjacencyTensor, [ 1-mu1, mu1 ], [ 1-nu1, nu1 ], P, Q )
                accuracies[ dummy].append( computeAccuracy( nodesLabels, labelsPred, K = K ) )
            
            mean_accuracy[ dummy ].append( np.mean( accuracies[ dummy] ) )
            ste_accuracy[ dummy ].append( np.std( accuracies[ dummy] ) / np.sqrt( nAverage ) )
        
    return ( mean_accuracy, ste_accuracy )


def plotFigure( x, accuracy_mean, accuracy_err, methods, 
               xticks = None, yticks = None, step = 1,
               xlabel = "Number of time step", ylabel = "Accuracy",
               legend_title = '$Q_{11}$',
               savefig = False, fileName = "fig.eps",
               legend_label = None ):
    
    if legend_label == None:
        legend_label = dict( )
        for method in methods:
            legend_label[ method ] = str(method)
    
    T = len( x )
    x = np.asarray( x )
    for method in methods:
        accuracy_mean[ method ] = np.asarray( accuracy_mean[ method ] )
        accuracy_err[ method ] = np.asarray( accuracy_err[ method ] )
        
    for method in methods:
        plt.errorbar( x[ : T : step], accuracy_mean[ method ][ : T : step ], yerr = accuracy_err[ method ][ : T : step ], linestyle = '-.', label = str( legend_label[ method ] ) )

    plt.xlabel( xlabel, fontsize = SIZE_LABELS )
    plt.ylabel( ylabel, fontsize = SIZE_LABELS )
    
    if xticks != None:
        plt.xticks( xticks, fontsize = SIZE_TICKS)
    else:
        plt.xticks( fontsize = SIZE_TICKS)
    
    if yticks != None:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
    else:
        plt.yticks( fontsize = SIZE_TICKS )
    
    if legend_title != None:
        legend = plt.legend(title = legend_title, loc='best',  fancybox = True, fontsize = SIZE_LEGEND )
        plt.setp( legend.get_title( ), fontsize = SIZE_LEGEND )
    else:
        legend = plt.legend( loc='best',  fancybox = True, fontsize = SIZE_LEGEND )
        plt.setp( legend.get_title( ), fontsize = SIZE_LEGEND )
    
    if( savefig ):
        plt.savefig( fileName, format='eps', bbox_inches='tight' )
    plt.show( )



# =============================================================================
# NEEDED FUNCTIONS
# =============================================================================

def makeTransitionMatrix( stationnaryDistribution, linkPersistence ):
    """
    Compute the transition matrix of a binary Markov Chain, 
    given the stationary distribution and the probability of transition 1 \to 1
    """
    p = stationnaryDistribution[1]
    P = np.zeros( (2,2) )
    P[1,1] = linkPersistence
    P[1,0] = 1 - linkPersistence
    P[0,1] = p * ( 1-linkPersistence) / (1-p)
    P[0,0] = 1 - P[0,1]
    return P


def computeAccuracy( labels_true, labels_pred, K = 2 ):
    return max( accuracy_score( labels_true, labels_pred ) , 1-accuracy_score( labels_true, labels_pred ) )


def followAccuracy( labels_true, labelsPred, K = 2, useTqdm = False ):
    """
    This function compute the accuracy of the labelling labelsPred at each timestep t
    Careful: only implemented for 2 cluster.
    
    """
    accuracy = []
    if useTqdm:
        loop = tqdm( range( labelsPred.shape[ 1 ] ) )
    else:
        loop = range( labelsPred.shape[ 1 ] )
        
    for t in loop:
        accuracy.append( computeAccuracy( labels_true, labelsPred[ :, t ] )  )

    return accuracy



# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def compareInitialization( mu1, nu1, P, Q, N, T, n_average = 10, savefig = False, filename = "fig.eps"):
    
    nodesLabels = np.ones( N, dtype = int )
    for i in range( N//2, N ):
        nodesLabels[ i ] = 2
    
    accuracy_RG = np.zeros( ( n_average, T ) )
    accuracy_SC = np.zeros( ( n_average, T ) )
    
    transitionRateMatrix = np.array( [ [ P, Q ], [ Q, P ] ] )
    initialDistributionRateMatrix = np.array( [ [ mu1, nu1 ], [ nu1, mu1 ] ] )

    for i in tqdm( range( n_average ) ):
        random.shuffle( nodesLabels ) #shuffle just to be sure the ordering doesn't matter
        ##Create the adjacency matrix
        adjacencyTensor = MarkovSBM.makeMDSBMAdjacencyMatrix( N, T, initialDistributionRateMatrix, transitionRateMatrix, nodesLabels )
        
        labelsPred = onlineLikelihood.onlineLikelihoodClustering_knownParameters ( adjacencyTensor, initialDistributionRateMatrix, transitionRateMatrix, K = 2, initialisation = 'random' )
        accuracy_RG[i,:] = followAccuracy( nodesLabels, labelsPred )
        
        labelsPred = onlineLikelihood.onlineLikelihoodClustering_knownParameters ( adjacencyTensor, initialDistributionRateMatrix, transitionRateMatrix, K = 2, initialisation = 'spectral clustering' )
        accuracy_SC[i,:] = followAccuracy( nodesLabels, labelsPred )
    
    mean_accuracy_RG = np.zeros( T )
    ste_accuracy_RG = np.zeros( T )
    mean_accuracy_SC = np.zeros( T )
    ste_accuracy_SC = np.zeros( T )
    
    for t in range(T):
        mean_accuracy_RG[t] = np.mean( accuracy_RG[ :, t ] )
        mean_accuracy_SC[t] = np.mean( accuracy_SC[ :, t ] )
        ste_accuracy_RG[t] = np.std( accuracy_RG[ :, t ] ) / np.sqrt( n_average )
        ste_accuracy_SC[t] = np.std( accuracy_SC[ :, t ] ) / np.sqrt( n_average )
        
    plt.errorbar( list( range( T ) ), mean_accuracy_SC, yerr = ste_accuracy_SC, linestyle = '--', label='Spectral Clustering' )
    plt.errorbar( list( range( T ) ), mean_accuracy_RG, yerr = ste_accuracy_RG, linestyle = '-.', label='Random Guessing' )
    legend = plt.legend( title = "Initialisation by:", loc=4,  fancybox=True, fontsize= SIZE_LEGEND )
    plt.setp( legend.get_title( ),fontsize= SIZE_LEGEND )
    plt.xlabel( "Number of time steps", fontsize = SIZE_LABELS )
    plt.xticks( np.arange(0, T, 5), fontsize= SIZE_TICKS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    #plt.xticks( np.arange(0,T,2), fontsize= SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    #plt.title("Evolution of accuracy, \n averaged over %s MSBM with n = %s, T = %s \n and parameters cin = %s, cout = %s, rin = %s and rout = %s" % (nAverage, n, T, cin, cout, rin, rout))
    #plt.savefig('comparison_initialization_cin_4,0_cout_1,5_qin_0,7_qout_0,3.eps', format='eps')

    if ( savefig ):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    plt.show()

    return ( accuracy_SC, accuracy_RG )



def compareKnowingNotKnowingModelParameters( mu1, nu1, P, Q, N, T, 
                                             n_average = 10,
                                             savefig = False, 
                                             filename = 'fig.eps',
                                             initialisation = 'random guess' ):
    nodesLabels = np.ones( N, dtype = int )
    for i in range( N // 2, N ):
        nodesLabels[ i ] = 2
    
    accuracy_knowing = np.zeros( ( n_average,T ) )
    accuracy_not_knowing = np.zeros( ( n_average,T ) )
    
    transitionRateMatrix = np.array( [ [ P, Q ], [ Q, P ] ] )
    initialDistributionRateMatrix = np.array( [ [ mu1, nu1 ] , [ nu1, mu1 ] ] )

    for i in tqdm( range( n_average ) ):
        random.shuffle( nodesLabels ) #shuffle just to be sure the community order doesn't matter
        adjacencyTensor = MarkovSBM.makeMDSBMAdjacencyMatrix( N, T, initialDistributionRateMatrix, transitionRateMatrix, nodesLabels )
        
        labelsPred = onlineLikelihood.onlineLikelihoodClustering_knownParameters ( adjacencyTensor, initialDistributionRateMatrix, transitionRateMatrix, K = 2, initialisation = initialisation )
        accuracy_knowing[ i, : ] = followAccuracy( nodesLabels, labelsPred, K=2 )
        
        ( labelsPred, predicted_P, predicted_Q ) = onlineLikelihood.onlineLikelihoodClustering_unkownParameters( adjacencyTensor, K = 2, initialisation = initialisation )
        accuracy_not_knowing[ i, : ] = followAccuracy( nodesLabels, labelsPred, K=2 )
    
    mean_accuracy_knowing = np.zeros( T )
    ste_accuracy_knowing = np.zeros( T )
    mean_accuracy_notknowing = np.zeros( T )
    ste_accuracy_notknowing = np.zeros( T )
    
    for t in range( T ):
        mean_accuracy_knowing[ t ] = np.mean( accuracy_knowing[ :, t ] )
        mean_accuracy_notknowing[ t ] = np.mean( accuracy_not_knowing[ :, t ] )
        ste_accuracy_knowing[ t ] = np.std( accuracy_knowing[ :, t ] ) / np.sqrt( n_average )
        ste_accuracy_notknowing[ t ] = np.std( accuracy_not_knowing[ :, t ] ) / np.sqrt( n_average )
    
    plt.errorbar( list( range( T ) ), mean_accuracy_knowing, yerr = ste_accuracy_knowing, linestyle = '--', label = 'Algorithm 2' )
    plt.errorbar( list( range( T ) ), mean_accuracy_notknowing, yerr = ste_accuracy_notknowing, linestyle = '-.', label = 'Algorithm 3' )
    
    legend = plt.legend( loc=4,  fancybox=True, fontsize= SIZE_LEGEND )
    plt.setp( legend.get_title( ),fontsize= SIZE_LEGEND )

    plt.xlabel( "Number of time steps", fontsize = SIZE_LABELS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( fontsize = SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    if( savefig ):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    else:
        plt.title( "Evolution of accuracy, \n averaged over %s MSBM with n = %s, T = %s \n and parameters muin = %s, muout = %s, Pin(1,1) = %s and Pout(1,1) = %s" % ( n_average, N, T, mu1, nu1, P[1,1], Q[1,1] ) )
    plt.show( )
    
    return ( accuracy_knowing, accuracy_not_knowing )