#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:18:59 2020

@author: mdreveto
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

import MarkovSBM as MarkovSBM
import online_likelihood_clustering_known_parameters as online_algo


SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18



"""
# =============================================================================
# To plot Fig 1 : theoretical T needed to get exact recovery 
# =============================================================================

n = 1000
cin = 4.0
cout = 1.5
muin, muout = cin  * np.log(n)/n , cout  * np.log(n)/n
TneededToGetExactRecovery = balayageToGetTofThreshold(muin, muout, n, K=2)
plotThreshold( TneededToGetExactRecovery )



# =============================================================================
# To plot Fig2 : greyscale plot of empirical accuracy, after T snapshots, as a function of Pin(1,1) and Pout(1,1)
# =============================================================================

T = 10
n = 500
cin = 4.0
cout = 1.5
muin, muout = cin  * np.log(n)/n , cout  * np.log(n)/n
qrange = [ i/50 for i in range(51) ]
accuracy = resultLikelihoodAlgorithm( n, T, muin, muout, q = qrange, initialisation = "RandomGuessing" )
plotAccuracy2D( accuracy, n, T, cin, cout, q = qrange, savefig = False, filename = "accuracy_SC_n_500_T_10_cin_4,0_cout_1,5.eps" )

"""



# =============================================================================
# Function to compute theoretical values 
# =============================================================================


def computeThreshold(muin, muout, Pin, Pout, T, n, K=2):
    I0 = (np.sqrt( muin ) - np.sqrt( muout ) )**2
    I1 = ( np.sqrt( Pin[0,1] ) - np.sqrt( Pout[0,1] ) )**2
    rho = 1 - np.sqrt( Pin[1,0] * Pout[1,0] ) / (1 - np.sqrt( Pin[1,1] * Pout[1,1] ) )
    I2 = 2 * rho * np.sqrt( Pin[0,1] * Pout[0,1] )
    I3 = 2 * rho * ( np.sqrt( muin * muout ) - np.sqrt( Pin[0,1] * Pout[0,1] ) / ( 1 - np.sqrt(Pin[1,1] * Pout[1,1] ) )  )
    return n/(np.log(n) * K) * (I0 + (T-1) * (I1+I2) + I3 * ( 1 - np.sqrt( Pin[1,1] * Pout[1,1] )**T ) )


def findTSuchThatExactRecoveryIsPossible(muin, muout, Pin, Pout, n, K = 2):
    T = 1
    threshold = ( np.sqrt( muin ) - np.sqrt( muout ) )**2
    I1 = ( np.sqrt( Pin[0,1] ) - np.sqrt( Pout[0,1] ) )**2
    lambdaa = np.sqrt( Pin[1,1] * Pout[1,1] )
    rho = 1 - np.sqrt( Pin[1,0] * Pout[1,0] ) / (1 - lambdaa )
    I2 = 2 * rho * np.sqrt( Pin[0,1] * Pout[0,1] )
    I3 = 2 * rho * ( np.sqrt( muin * muout ) - np.sqrt( Pin[0,1] * Pout[0,1] ) / ( 1 - lambdaa )  ) * (1 - lambdaa)
    J = I1 + I2

    while( n / (K * np.log(n) ) * threshold < K):
        T = T+1
        threshold = threshold + J + I3 * lambdaa**(T-2)
    return T


def balayageToGetTofThreshold(muin, muout, n, K=2, q = [i/300 for i in range(301)]):
    TneededToGetExactRecovery = np.zeros((len(q),len(q)))
    
    for i in tqdm(range(len(q) ) ): #Those 2 lines just to get the tqdm
        qin = q[i]
        Pin = MarkovSBM.makeTransitionMatrix( [1 - muin, muin] , qin)
        
        for qout in q:
            Pout = MarkovSBM.makeTransitionMatrix( [1 - muout, muout] , qout)
            if (qin == 1 and qout == 1):
                TneededToGetExactRecovery[q.index(qin), q.index(qout)] = 500000
                print('static case')
            if (muin == muout and qin == qout):
                TneededToGetExactRecovery[q.index(qin), q.index(qout)] = 500000
                print( 'impossible case' )
            else:
                TneededToGetExactRecovery[q.index(qin), q.index(qout)] = findTSuchThatExactRecoveryIsPossible(muin, muout, Pin, Pout, n, K = K )
    return TneededToGetExactRecovery




def plotThreshold(TneededToGetExactRecovery, savefig = False, filename = 'threshold_cin_?_cout_?.eps'):
    r = [ i/len( TneededToGetExactRecovery[:,0] ) for i in range( len(TneededToGetExactRecovery[:,0] ) +1 ) ]
    
    TneededToGetExactRecovery[-1,-1 ] = 2.33366e+07 #Just to avoid having +infinity
    plt.imshow(np.log10(TneededToGetExactRecovery), cmap = "binary")
    plt.imshow(np.log10(TneededToGetExactRecovery), extent =(min(r), max(r), min(r), max(r)), aspect = 'auto', cmap = 'binary', interpolation = 'none', origin = 'lowest')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=18)
    plt.ylabel( 'Pin(1,1)', fontsize=24 )
    plt.xlabel( 'Pout(1,1)', fontsize= 24 )
    plt.title( 'log(T) for exact recovery', fontsize= 24 )
              #, \n for mu_in = %s log(n)/n, mu_out = %s * log(n)/n' % (cin, cout))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    if (savefig):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    plt.show()

    return 


##########################

def resultLikelihoodAlgorithm( N, T, muin, muout, q = [i/50 for i in range(51)], initialisation = "SpectralClustering" ):
    
    initialDistributionRateMatrix = np.array( [ [ muin, muout ] , [ muout, muin ] ] )

    nodesLabels = np.zeros( N )
    
    for i in range( N//2 ):
        nodesLabels[i]= 1
    nodesLabels = nodesLabels.astype(int)
        
    accuracy = np.zeros((len(q),len(q)))
    
    q[0] = 0.01
    q[-1] = 0.99
    
    for i in tqdm( range(len(q) ) ): #Those 2 lines just to get the tqdm
        qin = q[i]
        Pin = MarkovSBM.makeTransitionMatrix( [ 1-muin, muin ] , qin )
        
        for qout in q:
            Pout = MarkovSBM.makeTransitionMatrix( [ 1-muout, muout ] , qout )
            TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )
            MSSBM_adja = MarkovSBM.makeMDSBMAdjacencyMatrix( N, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels )
            labelsPred_old = online_algo.likelihoodClustering( MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix, initialisation= initialisation )
            accuracy[q.index(qin), q.index(qout)] = online_algo.followAccuracy( nodesLabels, labelsPred_old )[T-1]
    return accuracy


def plotAccuracy2D(accuracy, n, T, cin, cout, q = [ i/50 for i in range(51) ], savefig = True, filename='accuracy_SC_n_?_T_?_cin_?_cout_?.eps.eps'  ):
    #plt.imshow(accuracy, cmap = "binary")
    plt.imshow(accuracy, extent = (0.0, 1.0, 0.0, 1.0), aspect = 'auto', cmap = 'binary_r', interpolation = 'none', origin = 'lowest')
    cbar = plt.colorbar( )
    cbar.ax.tick_params( labelsize=18 )
    plt.ylabel( 'Pin(1,1)', fontsize=24 )
    plt.xlabel( 'Pout(1,1)', fontsize=24 )
    plt.xticks(fontsize = SIZE_TICKS)
    plt.yticks(fontsize = SIZE_TICKS)
#    plt.title('Accuracy, n = %s, T = %s, cin = %s, cout = %s' % (n, T, cin, cout), fontsize= 24)
    if (savefig):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    plt.show()





