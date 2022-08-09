#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:08:37 2021

@author: mdreveto
"""

import numpy as np
import scipy as sp
from tqdm import tqdm 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


import two_steps_algo as two_steps_algo
import general_sbm_generator as sbm

SIZE_TITLE = 24
SIZE_LABELS = 24
SIZE_TICKS = 18
SIZE_LEGEND = 18


"""
# =============================================================================
# VARYING DISTRIBUTIONS : Figure 1
# =============================================================================

methods = [ 'Algorithm 1', 'XJL20' ]
N = 400
block_sizes = [ N//2, N//2 ]
nAverage = 25
p = 1
q = 1

nBins = int( 0.4*( np.log(np.log(N) ) )**4 )  #To use for continuous interactions
nBins = 40 # To use for discrete interactions

---------

distribution_type = 'gaussian'
parameter_f = 1 #Corresponds to the std of f
parameter_g_range = [ 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2 ] #std of g to test
xlabel = "tau"
fileName = 'accuracy_Normal_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_sigma_' + str(parameter_f) + '_nAverage_' + str( nAverage) + '.eps'
xticks = [ 0.8, 0.9, 1, 1.1, 1.2 ]

----------------

distribution_type = 'exponential'
parameter_f = 1 #Corresponds to lambda (parameter of exponential distrib  f)
parameter_g_range = [ 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3 ] #parameter of exponential distrib g to test
xlabel = "mu"
fileName = 'accuracy_zeroInflatedExponential_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.eps'

---------

distribution_type = 'pareto'
parameter_f = 2 #Pareto parameter of first distribution
parameter_g_range = [ 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4 ] #Pareto parameter of second distribution
xlabel = "b"
fileName = 'accuracy_Pareto_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.eps'

------------------------

distribution_type = 'gaussian mixture'
parameter_f = 1 #
parameter_g_range = [ 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75 ]
xlabel = "m"
fileName = 'accuracy_NormalAndNormalMixture_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.eps'
xticks = [ 0, 0.5, 1, 1.5 ]

-------------------

distribution_type = 'geometric'
parameter_f = 0.2 #Geometric parameter of first distribution
parameter_g_range = [ 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26 ] #geometric parameter of second distribution
xlabel = "b"
fileName = 'accuracy_Geometric_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.eps'
xticks = [ 0.16, 0.18, 0.20, 0.22, 0.24, 0.26 ]

-------------------

distribution_type = 'poisson'
parameter_f = 10 #Geometric parameter of first distribution
parameter_g_range = [ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 ] #geometric parameter of second distribution
parameter_g_range = np.linspace(8, 12, num=17)
xlabel = "mu"
fileName = 'accuracy_Poisson_N_' + str(N) + '_p_' + str(p) + '_q_' + str(q) + '_nAverage_' + str( nAverage) + '.eps'
xticks = [ 8, 9, 10, 11, 12 ]

---------------


accuracy_mean, accuracy_ste = accuracy_varying_parameter( block_sizes, p, q, 
                                                         parameter_f, parameter_g_range, 
                      distribution_type = distribution_type, nAverage = nAverage, 
                      methods = methods,
                      nBins = nBins )

ylabel = "Accuracy"
yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plotFigure( parameter_g_range, accuracy_mean, accuracy_ste, methods, 
               xticks = xticks, yticks = yticks,
               xlabel = xlabel, ylabel = "Accuracy",
               savefig = False, fileName = fileName )





# =============================================================================
# VARYING NETWORK SIZE : Figure 3
# =============================================================================

methods = [ 'Algorithm 1', 'XJL20' ]
nAverage = 25
p = 0.5
q = 0.5

----------------

sigma = 1.1
f = sbm.zeroInflatedNormal( p, 0, 1 )
g = sbm.zeroInflatedNormal( q, 0, sigma )
fileName = 'accuracy_varyingN_zeroInflatedNormal_p_' + str(p) + '_q_' + str(q) + '_sigma_' + str(sigma) + '_nAverage_' + str( nAverage) + '.eps'
N_range = [ 400, 600, 800, 1000, 1200, 1400 ]

----------------

m = 1.3
f = sbm.zeroInflatedNormal( p, 0, np.sqrt( m**2+1 ) )
g = sbm.zeroInflatedNormalMixture( q, m, 1 )
fileName = 'accuracy_varyingN_zeroInflatedNormalAndNormalMixture_p_' + str(p) + '_q_' + str(q) + '_m_' + str(m) + '_nAverage_' + str( nAverage) + '.eps'
N_range = [ 400, 600, 800, 1000, 1200, 1400, 1600, 1800 ]

----------------


accuracy_mean, accuracy_ste = varyingNetworkSize( f, g, N_range = N_range,
                       methods = methods,
                       nAverage = nAverage )

xticks = N_range
xticks = [ 400, 800, 1200, 1600 ]
yticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plotFigure( N_range, accuracy_mean, accuracy_ste, methods, 
               xticks = xticks, yticks = yticks,
               xlabel = "N", ylabel = "Accuracy",
               savefig = False, fileName = fileName )


"""



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
        plt.savefig( fileName, format='eps', bbox_inches='tight' )
    plt.show( )



def severalRuns( block_sizes, kernels, methods, nAverage = 10, 
                zeroElements = [0], nBins = 10 ):

    accuracies = dict( )
    for method in methods:
        accuracies[ method ] = [ ]
    
    f = kernels[ 0,0 ]
    g = kernels[ 0,1 ]
    
    for run in range( nAverage ):
        A, labels_true = sbm.generateGeneralSBM( block_sizes, kernels )
        for method in methods:
            if 'spectral clustering' in methods:
                Abinary = two_steps_algo.make_binary_matrix( A )
                labels_pred = two_steps_algo.spectralClustering( Abinary, K = len( block_sizes ) )
                accuracies['spectral clustering'].append( max( accuracy_score( labels_true, labels_pred ), 1 - accuracy_score( labels_true, labels_pred ) ) )
            else:
                labels_pred = two_steps_algo.two_step_clustering_general_interactions( A, f, g, K = len( block_sizes), initialisationMethod = method, nBins = nBins, zeroElements = zeroElements )
                accuracies[ method ].append( max( accuracy_score( labels_true, labels_pred ), 1 - accuracy_score( labels_true, labels_pred ) ) )
    
    return accuracies


def accuracy_varying_parameter( block_sizes, p, q, parameter_f, parameter_g_range, 
                      distribution_type = 'gaussian', nAverage = 10, 
                      methods = [ 'Algorithm 1', 'XJL20' ],
                      nBins = 10):
    
    accuracy_mean = dict( )
    accuracy_ste = dict( )
    for method in methods:
        accuracy_mean[ method ] = [ ]
        accuracy_ste[ method ] = [ ]
    
    for i in tqdm( range( len( parameter_g_range ) ) ):
        parameter_g = parameter_g_range[ i ]
        
        if distribution_type.lower() == 'gaussian':
            f = sbm.zeroInflatedNormal( p, 0, parameter_f )
            g = sbm.zeroInflatedNormal( q, 0, parameter_g )
            
        elif distribution_type.lower() == 'exponential':
            f = sbm.zeroInflatedGamma( p, 1, 0, 1 / parameter_f)
            g = sbm.zeroInflatedGamma( q, 1, 0, 1 / parameter_g )
        
        elif distribution_type.lower() == 'pareto':
            f = sbm.zeroInflatedPareto( p, parameter_f )
            g = sbm.zeroInflatedPareto( q, parameter_g )
        
        elif distribution_type.lower() == 'gaussian mixture':
            f = sbm.zeroInflatedNormal( p, 0, np.sqrt( parameter_g**2+1 ) )
            g = sbm.zeroInflatedNormalMixture( q, parameter_g, 1 )
            
        elif distribution_type.lower() == 'geometric':
            f = sbm.zeroInflatedGeometric( p, parameter_f )
            g = sbm.zeroInflatedGeometric( q, parameter_g )
        
        elif distribution_type.lower() == 'poisson':
            f = sbm.zeroInflatedPoisson( p, parameter_f )
            g = sbm.zeroInflatedPoisson( q, parameter_g )
        
        kernels = np.array( [ [f,g], [g,f] ] )
    
        zeroElements = getSetOfZerosElements( f, g )
        
        accuracies = severalRuns( block_sizes, kernels, methods = methods, nAverage = nAverage, zeroElements = zeroElements, nBins = nBins )
        for method in methods:
            accuracy_mean[ method ].append( np.mean( accuracies[ method ] ) )
            accuracy_ste[ method ].append( np.std( accuracies[ method ] ) / np.sqrt( nAverage ) )
    
    return accuracy_mean, accuracy_ste


def varyingNetworkSize( f, g, N_range = [ 500, 1000, 1500, 2000 ],
                       methods = [ 'Algorithm 1', 'XJL20' ],
                       nAverage = 10 ):

    accuracy_mean = dict( )
    accuracy_ste = dict( )
    for method in methods:
        accuracy_mean[ method ] = [ ]
        accuracy_ste[ method ] = [ ]
    
    zeroElements = getSetOfZerosElements( f, g )
    kernels = np.array( [ [f,g], [g,f] ] )
    
    for i in tqdm( range( len( N_range ) ) ):
        N = N_range[ i ]
        print( 'Number of nodes : ', N )
        block_sizes = [ N//2, N//2 ]
        nBins = int( 0.4*( np.log(np.log(N) ) )**4 )
        accuracies = severalRuns( block_sizes, kernels, methods = methods, nAverage = nAverage, zeroElements = zeroElements, nBins = nBins )
        for method in methods:
            accuracy_mean[ method ].append( np.mean( accuracies[ method ] ) )
            accuracy_ste[ method ].append( np.std( accuracies[ method ] ) / np.sqrt( nAverage ) )
    
    return accuracy_mean, accuracy_ste




def getSetOfZerosElements( f, g ):
    
    if f.dataType() == 'real valued':
        x0 = 1
        F = lambda x : f.cdf( x ) - f.cdf( -x )
        G = lambda x : g.cdf( x ) - g.cdf( -x )
        func = lambda x : Zhalf_bernoulli( F( x ), G( x ) )
        x = sp.optimize.fmin( func, x0 )[ 0 ]
    
        return [ -x, x ]
    
    elif f.dataType() == 'discrete':
        F = f.mass_function( 0 ) 
        G = g.mass_function( 0 )
        Z12 = Zhalf_bernoulli( F, G )
        
        continue_while = True
        x=1
        while continue_while and x <= 200:
            F += f.mass_function( x ) 
            G += g.mass_function( x )
            if Zhalf_bernoulli( F, G ) < Z12:
                Z12 = Zhalf_bernoulli( F, G )
                x += 1
            else:
                continue_while = False
        
        #print( 'The number max of labels is ', x )
        return [ i for i in range(x) ]
    
    else:
        raise TypeError ( 'There is a mistake in the function getSetOfZerosElements which computes the set of zero elements for Algorithm 1' )

def Zhalf_bernoulli( a, b ):
    return np.sqrt( a * b ) + np.sqrt( (1-a) * (1-b) )

