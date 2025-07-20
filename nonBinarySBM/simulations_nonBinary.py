#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:09:17 2023

@author: maximilien
"""


import numpy as np
import scipy as sp
from tqdm import tqdm 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

import general_sbm_generator as sbm

import nonBinary_clustering as nbc
import nonBinary_clustering_XJL as nbcXJL

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 20
SIZE_LEGEND = 18


"""
N = 500
block_sizes = [ N//2, N//2 ]
p = 0.5 
q = 0.5 
m = 1.3
f = sbm.zeroInflatedNormal( p, 0, np.sqrt( m**2+1 ) )
g = sbm.zeroInflatedNormalMixture( q, m, 1 )

kernel = np.array( [ [f,g], [g,f] ] )
X, labels_true = sbm.generateGeneralSBM( block_sizes, kernel )


# =============================================================================
# VARYING DISTRIBUTIONS: Figures 1 and 2
# =============================================================================


methods = [ 'Algorithm 1', 'Algorithm 2', 'XJL20' ]


### Figure 1

N = 200
n_clusters = 4
block_sizes = [ N // n_clusters for k in range( n_clusters ) ]
nAverage = 10
p = 1
q = 1

nBins = int( 0.4*( np.log(np.log(N) ) )**4 )  #To use for continuous interactions
#nBins = 40 # To use for discrete interactions



################# Figure 1a (dense Gaussian)

phis = [ lambda x: x, lambda x : x**2 ]

distribution_type = 'gaussian'
nBins = int( 0.4*( np.log(np.log(N) ) )**4 )  #To use for continuous interactions for XJL Algo

std_f = 1 #Corresponds to the std of f
std_g_range = np.linspace( 0.5, 2, num = 21) #std of g to test

f = sbm.zeroInflatedNormal( p, 0, std_f )
kernels = [ ]
for std_g in std_g_range:
    g = sbm.zeroInflatedNormal( q, 0, std_g )
    kernel = makeSymetricKernels( f, g, n_clusters )
    kernels.append( kernel )

xlabel = r"$\tau$"
fileName = 'accuracy_Normal_varying_std_g_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_sigma_' + str(std_f) + '_nAverage_' + str( nAverage) + '.pdf'
x = std_g_range
xticks = [ 0.5, 1, 1.5, 2 ]



################# Figure 1b (dense Geometric)

phis = [ lambda x: x ]

distribution_type = 'geometric'
nBins = 40 # To use for discrete interactions for XJL Algo

a_f = 0.2 #Corresponds to the parameter of f
#a_g_range = [ 0.1, 0.2, 0.3, 0.4 ] #std of g to test
a_g_range = np.linspace( 0.1, 0.5, num = 21 )

f = sbm.zeroInflatedGeometric( p, a_f )
kernels = [ ]
for a_g in a_g_range:
    g = sbm.zeroInflatedGeometric( q, a_g )
    kernel = makeSymetricKernels( f, g, n_clusters )
    kernels.append( kernel )

xlabel = r'${b}$'
fileName = 'accuracy_Geometric_varying_g_N' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_f_' + str(a_f) + '_nAverage_' + str( nAverage) + '.pdf'
x = a_g_range
xticks = [ 0.1, 0.2, 0.3, 0.4 , 0.5 ]




### Figure 2

N = 450
n_clusters = 3
block_sizes = [ N // n_clusters for k in range( n_clusters ) ]
nAverage = 10
p = 0.3
q = 0.3

nBins = int( 0.4*( np.log(np.log(N) ) )**4 )  #To use for continuous interactions
#nBins = 40 # To use for discrete interactions

################ Figure 2a (zero-inflated gaussian)

phis = [ lambda x: x, lambda x : x**2 ]

distribution_type = 'gaussian'
std_f = 1 #Corresponds to the std of f
std_g_range = np.linspace( 0.5, 2, num = 6) #std of g to test
std_g_range = np.linspace( 0.5, 2, num = 21) #std of g to test


f = sbm.zeroInflatedNormal( p, 0, std_f )
kernels = [ ]
for std_g in std_g_range:
    g = sbm.zeroInflatedNormal( q, 0, std_g )
    kernel = makeSymetricKernels( f, g, n_clusters )
    kernels.append( kernel )

xlabel = r"$\tau$"
fileName = 'accuracy_Normal_varying_std_g_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_sigma_' + str(std_f) + '_nAverage_' + str( nAverage) + '.pdf'
x = std_g_range
xticks = [ 0.5, 1, 1.5, 2 ]


################ Figure 2b (zero-inflated exponential)

phis = [ lambda x: x ]

parameter_f_range = np.linspace(0.4, 2.2, num=31)

distribution_type = 'exponential'
parameter_f_range = np.linspace(0.4, 2.2, num=31) #Corresponds to lambda (parameter of exponential distrib  f)
parameter_g = 1 #parameter of exponential distrib g to test

g = sbm.zeroInflatedGamma( p, 1, 0, 1 / parameter_g )
kernels = [ ]
for parameter_f in parameter_f_range:
    f = sbm.zeroInflatedGamma( q, 1, 0, 1 / parameter_f )
    kernel = makeSymetricKernels( f, g, n_clusters )
    kernels.append( kernel )

xlabel = r"$\lambda_f$"
x = parameter_f_range
xticks = [ 0.5, 1, 1.5, 2 ]
fileName = 'accuracy_zeroInflatedExponential_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.pdf'



##### To run and plot each figs (1a, 1b, 2a, 2b): 

accuracy_mean, accuracy_ste = accuracy_varying_parameter( block_sizes, kernels, nAverage = nAverage, 
                               methods = methods, phis = phis, 
                               distribution_type = distribution_type )

ylabel = "Accuracy"
yticks = [0.25, 0.5, 0.75, 1.0]

plotFigure( x, accuracy_mean, accuracy_ste, methods, 
               xticks = xticks, yticks = yticks,
               xlabel = xlabel, ylabel = "Accuracy",
               savefig = False, fileName = fileName )







---------



---------

distribution_type = 'gaussian'
mean_f_range = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
mean_g = 0
std = 1

g = sbm.zeroInflatedNormal( p, mean_g, std )
kernels = [ ]

for mean_f in mean_f_range:
    f = sbm.zeroInflatedNormal( q, mean_f, std )
    kernels.append( np.array( [ [f,g], [g,f] ] ) )

xlabel = r"$\mu_{f}$"
fileName = 'accuracy_Normal_varying_mean_f_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_sigma_' + str(std) + '_nAverage_' + str( nAverage) + '.eps'
x = mean_f_range
xticks = [-0.4, -0.2, 0, 0.2, 0.4 ]


----------------

distribution_type = 'exponential'
parameter_f_range = [ 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3 ] #Corresponds to lambda (parameter of exponential distrib  f)
parameter_g = 1 #parameter of exponential distrib g to test

g = sbm.zeroInflatedGamma( p, 1, 0, 1 / parameter_g )
kernels = [ ]
for parameter_f in parameter_f_range:
    f = sbm.zeroInflatedGamma( q, 1, 0, 1 / parameter_f )
    kernel = makeSymetricKernels( f, g, n_clusters )
    kernels.append( kernel )

xlabel = "lambda_f"
x = parameter_f_range
xticks = parameter_f_range
fileName = 'accuracy_zeroInflatedExponential_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.eps'


---------

distribution_type = 'gaussian_doublyGaussian'
m_range = [ 1, 2, 5, 10, 15 ]

kernels = [ ]
for m in m_range:
    f = sbm.zeroInflatedNormal( p, 0, np.sqrt( m**2+1 ) )
    g = sbm.zeroInflatedNormalMixture( q, m, 1 )
    kernels.append( np.array( [ [f,g], [g,f] ] ) )

xlabel = "m"
x = m_range
xticks = m_range
fileName = 'accuracy_zeroInflatedDoublyGaussian_N_' + str(N) + '_p_' + str( p ) + '_q_' + str( q ) + '_nAverage_' + str( nAverage ) + '.eps'


---------




---------







# =============================================================================
# VARYING NETWORK SIZE
# =============================================================================

methods = [ 'Algorithm 1', 'XJL20' ]
nAverage = 10
p = 0.4
q = 0.4
----------------

sigma = 1.2
f = sbm.zeroInflatedNormal( p, 0, 1 )
g = sbm.zeroInflatedNormal( q, 0, sigma )
fileName = 'accuracy_varyingN_zeroInflatedNormal_p_' + str(p) + '_q_' + str(q) + 'means_0_sigma_f_0_sigma_g_' + str( sigma ) + '_nAverage_' + str( nAverage ) + '.eps'
N_range = [ 400, 600, 800, 1000, 1200, 1400 ]
xticks = N_range

----------------

mean_f = 1
f = sbm.zeroInflatedNormal( p, mean_f, 1 )
g = sbm.zeroInflatedNormal( q, 0, 1 )
fileName = 'accuracy_varyingN_zeroInflatedNormal_p_' + str(p) + '_q_' + str(q) + 'mean_f_' + str(mean_f) + '_nAverage_' + str( nAverage ) + '.eps'
N_range = [ 400, 600, 800, 1000, 1200, 1400 ]
xticks = N_range

----------------

m = 1.3
f = sbm.zeroInflatedNormal( p, 0, np.sqrt( m**2+1 ) )
g = sbm.zeroInflatedNormalMixture( q, m, 1 )
fileName = 'accuracy_varyingN_zeroInflatedNormalAndNormalMixture_p_' + str(p) + '_q_' + str(q) + '_m_' + str(m) + '_nAverage_' + str( nAverage) + '.eps'
N_range = [ 400, 600, 800, 1000, 1200, 1400, 1600, 1800 ]
xticks = [ 400, 800, 1200, 1600 ]
----------------


accuracy_mean, accuracy_ste = varyingNetworkSize( f, g, N_range = N_range,
                       methods = methods,
                       nAverage = nAverage )

yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plotFigure( N_range, accuracy_mean, accuracy_ste, methods, 
               xticks = xticks, yticks = yticks,
               xlabel = "N", ylabel = "Accuracy",
               savefig = False, fileName = fileName )





"""
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


def makeSymetricKernels( f, g, n_clusters ):
    kernel = dict()
    for a in range( n_clusters ):
        for b in range( n_clusters ):
            if a == b:
                kernel[ a, b ] = f
            else:
                kernel[ a, b ] = g
    return kernel



def plotFigure( x, accuracy_mean, accuracy_err, methods, 
               xticks = None, yticks = None,
               xlabel = "x", ylabel = "Accuracy",
               savefig = False, fileName = "fig.eps" ):
    
    for method in methods:
        plt.errorbar( x, accuracy_mean[ method ], yerr = accuracy_err[ method ], linestyle = '-.', label = method )

    plt.xlabel( xlabel, fontsize = SIZE_LABELS)
    plt.ylabel( ylabel, fontsize = SIZE_LABELS)
    
    if xticks != None:
        plt.xticks( xticks, fontsize = SIZE_TICKS)
    else:
        plt.xticks( fontsize = SIZE_TICKS)
    
    if yticks != None:
        plt.yticks( yticks, fontsize = SIZE_TICKS )
    else:
        plt.yticks( fontsize = SIZE_TICKS )

    legend = plt.legend( loc=0,  fancybox = True, fontsize = SIZE_LEGEND )
    plt.setp( legend.get_title(),fontsize = SIZE_LEGEND )

    if(savefig):
        plt.savefig( fileName, bbox_inches='tight' )
    plt.show( )




def severalRuns( block_sizes, kernel, methods, phis = None, nAverage = 10, distribution_type = 'gaussian' ):
    
    n_clusters = len( block_sizes )
    
    #if kernel.shape[0]!= n_clusters and kernel.shape[1] != n_clusters:
    if len( kernel ) != n_clusters**2:
        raise TypeError( 'The size of the probability kernels does not math the number of clusters' )
    
    accuracies = dict( )
    for method in methods:
        accuracies[ method ] = [ ]
        
    for run in range( nAverage ):
        X, labels_true = sbm.generateGeneralSBM( block_sizes, kernel )
        for method in methods:
            if method == 'Algorithm 1':
                labels_pred = nbc.nonBinaryClustering( X, phis, n_clusters = n_clusters, interaction_distribution = distribution_type, improvement = True )
            elif method == 'Algorithm 2' or method == 'Algorithm 1 init' or method == 'aggregatingSpectralEmbeddings': 
                labels_pred = nbc.nonBinaryClustering( X, phis, n_clusters = n_clusters, improvement = False )
            elif method == 'XJL20':
                labels_pred = nbcXJL.XJL_clustering( X, n_clusters = n_clusters )
            else:
                raise TypeError( 'The algorithm is not implemented' )
            
            accuracies[ method ].append( computeAccuracy( labels_true, labels_pred ) )
            #accuracies[ method ].append( max( accuracy_score( labels_true, labels_pred ), 1-accuracy_score( labels_true, labels_pred ) ) )

    return accuracies



def accuracy_varying_parameter( block_sizes,  kernels, nAverage = 10, 
                               methods = [ 'Algorithm 1', 'XJL20' ], phis = None,
                               distribution_type = 'gaussian' ):
    
    accuracy_mean = dict( )
    accuracy_ste = dict( )
    for method in methods:
        accuracy_mean[ method ] = [ ]
        accuracy_ste[ method ] = [ ]
    
    for i in tqdm( range( len( kernels ) ) ):
        kernel = kernels[ i ]
        
        accuracies = severalRuns( block_sizes, kernel, methods = methods, phis = phis, nAverage = nAverage, distribution_type = distribution_type )
        for method in methods:
            accuracy_mean[ method ].append( np.mean( accuracies[ method ] ) )
            accuracy_ste[ method ].append( np.std( accuracies[ method ] ) / np.sqrt( nAverage ) )
    
    return accuracy_mean, accuracy_ste



def varyingNetworkSize( f, g, N_range = [ 500, 1000, 1500, 2000 ],
                       methods = [ 'Algorithm 1', 'XJL20' ],
                       nAverage = 10, phis = None ):

    accuracy_mean = dict( )
    accuracy_ste = dict( )
    for method in methods:
        accuracy_mean[ method ] = [ ]
        accuracy_ste[ method ] = [ ]
    
    kernels = np.array( [ [f,g], [g,f] ] )
    
    for i in tqdm( range( len( N_range ) ) ):
        N = N_range[ i ]
        #print( 'Number of nodes : ', N )
        block_sizes = [ N//2, N//2 ]
        accuracies = severalRuns( block_sizes, kernels, methods, phis = phis, nAverage = nAverage )
        for method in methods:
            accuracy_mean[ method ].append( np.mean( accuracies[ method ] ) )
            accuracy_ste[ method ].append( np.std( accuracies[ method ] ) / np.sqrt( nAverage ) )
    
    return accuracy_mean, accuracy_ste
