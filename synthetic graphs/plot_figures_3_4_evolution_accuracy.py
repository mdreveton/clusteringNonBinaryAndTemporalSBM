#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 15:35:26 2020

@author: mdreveto
"""


import numpy as np
import random as random
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
# To plot the comparison between initialization methods (RandomGuess vs SpectralClustering)
# =============================================================================


T = 13
n = 500
cin = 2.5
cout = 1.5
qin = 0.7 #qin repesent Pin(1,1)
qout = 0.3  #same as qin, but for outside community node pairs

muin, muout = cin  * np.log(n)/n , cout  * np.log(n)/n
Pin = MarkovSBM.makeTransitionMatrix( [1-muin, muin], qin )
Pout = MarkovSBM.makeTransitionMatrix( [1-muout, muout], qout )

#findTSuchThatExactRecoveryIsPossible(muin, muout, Pin, Pout, n, K = 2)
(accuracy_SC, accuracy_RG) = compareInitialization( muin, muout, Pin, Pout, n, T, n_average = 50, savefig = False, filename = "comparison_initialization_cin_2,5_cout_1,5_qin_0,7_qout_0,3.eps" )





# =============================================================================
# Figure4: Accuracy in the constant degree regime, for different Pout(1,1)
# =============================================================================

### Figure 4a

n = 500
T = 130
cin = 2.5
cout = 1.5

scenarios = [ ]
qoutTested = [ cout / n, 0.3, 0.6, 0.9 ]
for qout in qoutTested:
    scenarios.append( [ cin, cout, 0.6, qout ] )
n_average = 50
( mean_accuracy, ste_accuracy ) = plotComparisonDifferentScenario( n, T, scenarios, n_average, regime = "constant", 
                                    savefig = False, filename = 'accuracy_constant_degree_regime_n_500_in_2,5_cout_1,5_Pin_0,6.eps' )

Tthresh = 130
step = 2
for i in range(len(scenarios)):
    plt.errorbar( list( range(0, Tthresh, step) ), mean_accuracy[i,:Tthresh:step], yerr = ste_accuracy[i,:Tthresh:step], linestyle = '-.', label= str(scenarios[i][3]))
legend = plt.legend(title="Pout(1,1):", loc=4,  fancybox=True, fontsize= SIZE_LEGEND)
plt.setp(legend.get_title(),fontsize= SIZE_LEGEND )
plt.xlabel( "Number of time steps", fontsize = SIZE_LABELS )
plt.xticks( np.arange(0, T, 250), fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
plt.ylabel("Accuracy", fontsize = SIZE_LABELS )
plt.savefig( "accuracy_constant_degree_regime_n_500_in_2,5_cout_1,5_Pin_0,6.eps" , format='eps', bbox_inches='tight' )
plt.show()




### Figure 4b

n = 100
T = 2000
cin = 0.15
cout = 0.1

scenarios = [ ]
qoutTested = [ 0.1, 0.2, 0.3, 0.4 ]
for qout in qoutTested:
    scenarios.append( [ cin, cout, 0.4, qout ] )
n_average = 50
( mean_accuracy, ste_accuracy ) = plotComparisonDifferentScenario( n, T, scenarios, n_average, regime = "constant", 
                                    savefig = False, filename = 'accuracy_constant_degree_regime_n_100_cin_0,15_cout_0,1_Pin_0,4.eps' )



Tthresh = 2000
step = 20

for i in range(len(scenarios)):
    plt.errorbar( list( range(0, Tthresh, step) ), mean_accuracy[i,:Tthresh:step], yerr = ste_accuracy[i,:Tthresh:step], linestyle = '-.', label= str(scenarios[i][3]))
legend = plt.legend(title="Pout(1,1):", loc=4,  fancybox=True, fontsize= SIZE_LEGEND)
plt.setp(legend.get_title(),fontsize= SIZE_LEGEND )
plt.xlabel("Number of time steps", fontsize = SIZE_LABELS)
plt.xticks( np.arange(0, T, 400), fontsize = SIZE_TICKS )
plt.yticks( fontsize = SIZE_TICKS )
plt.ylabel("Accuracy", fontsize = SIZE_LABELS )
plt.savefig( "accuracy_constant_degree_regime_n_100_cin_0,15_cout_0,1_Pin_0,4.eps" , format='eps', bbox_inches='tight' )
plt.show()


"""

    

def compareInitialization(muin, muout, Pin, Pout, n, T, n_average = 10, savefig = True, filename = "comparison_initialization_cin_?_cout_?_qin_?_qout_?.eps"):
    nodesLabels = np.zeros(n)
    for i in range(n//2):
        nodesLabels[i]= 1
    nodesLabels = nodesLabels.astype(int)
    
    accuracy_RG = np.zeros( ( n_average,T ) )
    accuracy_SC = np.zeros( ( n_average,T ) )
    
    TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )
    initialDistributionRateMatrix = np.array( [ [muin, muout], [muout, muin] ] )

    for i in tqdm( range( n_average ) ):
        random.shuffle( nodesLabels ) #shuffle just to be sure the ordering doesn't matter
        ##Create the adjacency matrix
        MSSBM_adja = MarkovSBM.makeMDSBMAdjacencyMatrix( n, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels )
        
        labelsPred = online_algo.likelihoodClustering( MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix, initialisation = "RandomGuessing" )
        accuracy_RG[i,:] = online_algo.followAccuracy( nodesLabels, labelsPred )
        labelsPred = online_algo.likelihoodClustering( MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix, initialisation = "SpectralClustering" )
        accuracy_SC[i,:] = online_algo.followAccuracy( nodesLabels, labelsPred )
    
    mean_accuracy_RG = np.zeros( T )
    ste_accuracy_RG = np.zeros( T )
    mean_accuracy_SC = np.zeros( T )
    ste_accuracy_SC = np.zeros( T )
    
    for t in range(T):
        mean_accuracy_RG[t] = np.mean( accuracy_RG[ :, t ] )
        mean_accuracy_SC[t] = np.mean( accuracy_SC[ :, t ] )
        ste_accuracy_RG[t] = np.std( accuracy_RG[ :, t ] ) / np.sqrt( n_average )
        ste_accuracy_SC[t] = np.std( accuracy_SC[ :, t ] ) / np.sqrt( n_average )
        
    plt.errorbar( list( range(T) ), mean_accuracy_SC, yerr = ste_accuracy_SC, linestyle = '--', label='Spectral Clustering' )
    plt.errorbar( list( range(T) ), mean_accuracy_RG, yerr = ste_accuracy_RG, linestyle = '-.', label='Random Guessing' )
    legend = plt.legend( title= "Initialisation by:", loc=4,  fancybox=True, fontsize= SIZE_LEGEND )
    plt.setp( legend.get_title(),fontsize= SIZE_LEGEND )
    plt.xlabel( "Number of time steps", fontsize = SIZE_LABELS )
    plt.xticks( np.arange(0, T, 5), fontsize= SIZE_TICKS )
    plt.ylabel( "Accuracy", fontsize = SIZE_LABELS )
    plt.xticks( np.arange(0,T,2), fontsize= SIZE_TICKS )
    plt.yticks( fontsize = SIZE_TICKS )
    #plt.title("Evolution of accuracy, \n averaged over %s MSBM with n = %s, T = %s \n and parameters cin = %s, cout = %s, rin = %s and rout = %s" % (nAverage, n, T, cin, cout, rin, rout))
    #plt.savefig('comparison_initialization_cin_4,0_cout_1,5_qin_0,7_qout_0,3.eps', format='eps')

    if ( savefig ):
        plt.savefig( filename, format='eps', bbox_inches='tight' )
    plt.show()

    return ( accuracy_SC, accuracy_RG )




def plotComparisonDifferentScenario(n, T, scenarios, n_average = 10, regime = "constant", savefig = False, filename = 'accuracy' ):
    mean_accuracy = np.zeros( (len(scenarios), T) )
    ste_accuracy = np.zeros( (len(scenarios), T) )
    
    for scenarioNumber in tqdm(range(len(scenarios))):
        accuracy = np.zeros( (n_average,T) )
        cin = scenarios[scenarioNumber][0]
        cout = scenarios[scenarioNumber][1]
        qin = scenarios[scenarioNumber][2]
        qout = scenarios[scenarioNumber][3]
        paramMatrix = np.array( [ [cin,cout] , [cout , cin] ] )
        if (regime == 'constant'):
            initialDistributionRateMatrix = paramMatrix /n
        if (regime == "logarithmic"):
            initialDistributionRateMatrix = paramMatrix * np.log(n) /n
        
        Pin = MarkovSBM.makeTransitionMatrix( [ 1 - initialDistributionRateMatrix[0,0], initialDistributionRateMatrix[0,0] ], qin )
        Pout = MarkovSBM.makeTransitionMatrix( [1 - initialDistributionRateMatrix[0,1], initialDistributionRateMatrix[0,1] ], qout )
        TransitionRateMatrix = np.array( [ [Pin, Pout], [Pout, Pin] ] )

        for i in range(n_average):
            nodesLabels = np.zeros(n)
            for j in range(n):
                nodesLabels[j]= ( random.random() > 0.5 ) * 1
            nodesLabels = nodesLabels.astype(int)
            MSSBM_adja = MarkovSBM.makeMDSBMAdjacencyMatrix(n, T, initialDistributionRateMatrix, TransitionRateMatrix, nodesLabels)
            labelsPred = online_algo.likelihoodClustering(MSSBM_adja, initialDistributionRateMatrix, TransitionRateMatrix,  initialisation = "RandomGuessing")
            accuracy[i,:] = online_algo.followAccuracy(nodesLabels, labelsPred)
        for t in range(T):
            mean_accuracy[scenarioNumber,t] = np.mean(accuracy[:,t])
            ste_accuracy[scenarioNumber,t] = np.std(accuracy[:,t]) / np.sqrt( n_average )
    
    for i in range(len(scenarios)):
        plt.errorbar( list(range(T)), mean_accuracy[i,:], yerr = ste_accuracy[i,:], linestyle = '-.', label= 'Pout(1,1) = ' + str(scenarios[i][3]))
    legend = plt.legend(title="Model parameters:", loc=4,  fancybox=True, fontsize= SIZE_LEGEND)
    plt.setp(legend.get_title(),fontsize= SIZE_LEGEND)
    plt.xlabel("Number of time step", fontsize = SIZE_LABELS)
    plt.xticks( np.arange( 0, T, max( 1, T//5 ) ) , fontsize = SIZE_TICKS)
    plt.ylabel("Accuracy", fontsize = SIZE_LABELS)
    plt.yticks( fontsize = SIZE_TICKS )
    #plt.title("Evolution of accuracy, \n averaged over %s MSBM with n = %s, T = %s, \n cin = %s, cout = %s (constant degree regime) \, and Pin(1,1) = %s" % (n_average, n, T, cin, cout, scenarios[0][2]) , fontsize = SIZE_TITLE)
    plt.show()
    
    if (savefig):
        plt.savefig( filename, format='eps' )

    
    return (mean_accuracy, ste_accuracy)
