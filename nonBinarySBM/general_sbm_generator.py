#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:58:20 2021

@author: mdreveto
"""

import numpy as np
from tqdm import tqdm 
import scipy.stats as stats
import networkx as nx


"""
N = 1000
p = 0.02
q = 0.01
a = 0.5
b = 0.4

f = zeroInflatedGeometric( p, a )
g = zeroInflatedGeometric( q, b )
block_sizes = [ N//2, N//2 ]
kernels = np.array( [ [f,g], [g,f] ] )


A, labels_true = generateGeneralSBM( block_sizes, kernels  )
"""


"""
N = 1000
p = 0.02
q = 0.01
m = 10
s = 1
lambdaa = 1


f = zeroInflatedDoublyExponential( p, lambdaa )
g = zeroInflatedNormal( q, m, s )
block_sizes = [ N//2, N//2 ]
kernels = np.array( [ [f,g], [g,f] ] )

A, labels_true = generateGeneralSBM( block_sizes, kernels  )

"""



def generateGeneralSBM( block_sizes, kernels, tqdm_ = False ):
    """
    This generates a SBM with K = len( block_sizes ) blocks
    f = intra-community interaction kernel
    g = inter-community interaction kernel
    
    in_parameters should be
    """
    
    N = sum( block_sizes )
    K = len( block_sizes )
    community_labels = []
    
    for k in range( K ):
        community_labels += [ k+1] * block_sizes[ k ] 
    
    """
    if community_labels:
        np.random.shuffle( community_labels )
    """
    
    A = np.zeros( ( N, N ) )
    
    non_zero_probas = np.zeros( (K,K) )
    sparse = False
    for k in range( K ):
        for ell in range( K ):
            non_zero_probas[ k, ell ] =  kernels[k,ell].non_zero_proba
            if non_zero_probas[ k, ell ] < 0.7:
                sparse = True

    if sparse:
        Gbinary = nx.generators.community.stochastic_block_model( block_sizes, non_zero_probas )
        #Abinary = nx.adjacency_matrix( Gbinary )
        #non_zeros_indexes = Abinary.nonzero( )
        for edge in Gbinary.edges( ):
            node1 = edge[ 0 ]
            node2 = edge[ 1 ]
        #for i in range( len( non_zeros_indexes[0] ) ):
        #    node1 = non_zeros_indexes[0][i]
        #    node2 = non_zeros_indexes[1][i]
            proba_function = kernels[ community_labels[ node1 ] - 1, community_labels[ node2 ] - 1 ]
            A[ node1, node2 ] = proba_function.generate_nonzero_random_variable( )
            A[ node2, node1 ] = A[ node1, node2 ]
    else:
        if (tqdm_):
            loop = tqdm(range(N))
        else:
            loop = range( N )
            
        for i in loop:
            for j in range( i, N ):
                proba_function = kernels[ community_labels[i] - 1, community_labels[j] - 1 ]
                A[i,j] = proba_function.generate_random_variable( )
                A[j,i] = A[i,j]

    return A, community_labels



def bernoulli( p ):
    return lambda x : x*p + (1-x) * (1-p) if x in [0,1] else TypeError('x should be 0 or 1 to consider Bernoulli' ) 


def generateInflatedGeometric( p, a, distribution = 'geometric' ):
    
    if distribution == 'geometric':
        if a>1 or a<=0:
            raise TypeError('The geometric parameter should be in (0,1]' )
        else:
            return lambda x : 0 if  stats.bernoulli.rvs( p ) == 1 else stats.geom.rvs( a )
            #x = stats.geom.rvs( a ) 
    
    elif distribution == 'exponential':
        if a<0:
            raise TypeError('The exponential parameter should be in (0,infty)' )
        else:
            return lambda x : 0 if  stats.bernoulli.rvs( p ) == 1 else stats.expon.rvs(a)
    
    else:
        raise TypeError('This distribution is not supported')    



def generate_zeroInflatedDistributions( p, distrib_parameters ):
    distribution = distrib_parameters[0]
    if distribution == 'geometric':
        return zeroInflatedGeometric( p, distrib_parameters[1] )
    elif distribution == 'doubly exponential':
        return zeroInflatedDistributions( p, distrib_parameters[1] )
    elif distribution == 'normal':
        return zeroInflatedNormal( p, distrib_parameters[1], distrib_parameters[2] )
    else:
        raise TypeError( 'This distribution type is not supported' )



class zeroInflatedDistributions_new( ):
    
    def __init__( self, p, parameters, distributionType = 'gaussian' ):
        if p<0 or p>1:
            raise TypeError( 'parameter p should be in [0,1]' )
        self.non_zero_proba = p
        
        if distributionType == 'geometric':
            self.non_zero_distribution = stats.geom
            self.a = parameters[ 0 ]
        elif distributionType == 'gaussian':
            self.non_zero_distribution = stats.norm
            self.mean = parameters[ 0 ]
            self.std = parameters[ 1 ]
            
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return self.non_zero_distribution.rvs( self.geometric_parameter )




class zeroInflatedDistributions( ):
    
    def __init__( self, p ):
        if p<0 or p>1:
            raise TypeError( 'parameter p should be in [0,1]' )
        self.non_zero_proba = p


class zeroInflatedGeometric( zeroInflatedDistributions ):
    
    def __init__(self, p, a ):
        zeroInflatedDistributions.__init__( self, p )
        if a>1 or a<=0:
            raise TypeError('The geometric parameter should be in (0,1]' )
        self.geometric_parameter = a
    
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.geom.rvs( self.geometric_parameter )
    
    def generate_nonzero_random_variable( self ):
        return stats.geom.rvs( self.geometric_parameter )
    
    def mass_function( self, x ):
        return ( x==0 ) * ( 1 - self.non_zero_proba ) + self.non_zero_proba * stats.geom.pmf( x , self.geometric_parameter )
    
    def dataType( self ):
        return 'discrete'



class zeroInflatedPoisson( zeroInflatedDistributions ):
    
    def __init__(self, p, mu ):
        zeroInflatedDistributions.__init__( self, p )
        if mu <= 0:
            raise TypeError('The Poisson parameter should be positive' )
        self.poisson_parameter = mu
    
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.poisson.rvs( self.poisson_parameter )
    
    def generate_nonzero_random_variable( self ):
        return stats.poisson.rvs( self.poisson_parameter)
    
    def mass_function( self, x ):
        return ( x == 0 ) * ( 1 - self.non_zero_proba ) + self.non_zero_proba * stats.poisson.pmf( x , self.poisson_parameter )
    
    def dataType( self ):
        return 'discrete'



class zeroInflatedDoublyExponential( zeroInflatedDistributions ):
    
    def __init__(self, p, lambdaa ):
        zeroInflatedDistributions.__init__( self, p )
        if lambdaa <= 0:
            raise TypeError('The exponential parameter should be in (0,infty)' )
        self.exponential_parameter = lambdaa
    
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.expon.rvs( scale = self.exponential_parameter )
    
    def generate_nonzero_random_variable( self ):
        t = stats.expon.rvs( scale = self.exponential_parameter )
        if stats.bernoulli.rvs( 1/2 ) == 1:
            return t
        else:
            return - t
    
    def mass_function( self, x ):
        if x ==0 :
            return 1 - self.non_zero_proba
        else:
            return  self.non_zero_proba * 1 / 2 * self.exponential_parameter * np.exp( - self.exponential_parameter * np.abs( x ) )
            #return self.non_zero_proba * stats.expon.pdf( x , scale = self.exponential_parameter )

    def dataType( self ):
        return 'real valued'



class zeroInflatedNormal( zeroInflatedDistributions ):
    
    def __init__(self, p, m, sigma ):
        zeroInflatedDistributions.__init__( self, p )
        if sigma <= 0:
            raise TypeError('The standard deviation should be in (0,infty)' )
        self.mean = m
        self.std = sigma        
        
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.norm.rvs( loc= self.mean, scale = self.std )
    
    def generate_nonzero_random_variable( self ):
        return stats.norm.rvs( loc= self.mean, scale = self.std )    
    
    def mass_function( self, x ):
        return (x==0) * ( 1 - self.non_zero_proba ) + self.non_zero_proba * stats.norm.pdf( x , loc= self.mean, scale = self.std )
        """
        if x == 0 :
            return 1 - self.non_zero_proba + self.non_zero_proba * stats.norm.pdf( x , loc= self.mean, scale = self.std )
        else:
            return self.non_zero_proba * stats.norm.pdf( x , loc= self.mean, scale = self.std )
        """
    
    def mass_function_withoutZeroInflation( self, x ):
        return self.non_zero_proba * stats.norm.pdf( x , loc= self.mean, scale = self.std )
    
    def cdf( self, x ):
        return self.non_zero_proba * stats.norm.cdf( x, loc = self.mean, scale = self.std ) + (x>=0)*(1-self.non_zero_proba)
    
    def dataType( self ):
        return 'real valued'
    
    
    
class zeroInflatedNormalMixture( zeroInflatedDistributions ):
    
    def __init__(self, p, m, sigma ):
        zeroInflatedDistributions.__init__( self, p )
        if sigma <= 0:
            raise TypeError('The variance should be in (0,infty)' )
        self.mean = m
        self.std = sigma
        
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            if np.random.rand() > 1/2:
                return stats.norm.rvs( loc= self.mean, scale = self.std )
            else:
                return stats.norm.rvs( loc= - self.mean, scale = self.std )
    
    def generate_nonzero_random_variable( self ):
        if np.random.rand() > 1/2:
            return stats.norm.rvs( loc= self.mean, scale = self.std )
        else:
            return stats.norm.rvs( loc= - self.mean, scale = self.std )
    
    def mass_function( self, x ):
        return (x==0) * ( 1 - self.non_zero_proba ) + self.non_zero_proba *1/2 * ( stats.norm.pdf( x , loc= self.mean, scale = self.std ) + stats.norm.pdf( x , loc= -self.mean, scale = self.std ) )

    def cdf( self, x ):
        return self.non_zero_proba * 1/2 * ( stats.norm.cdf( x, loc = -self.mean, scale = self.std ) + stats.norm.cdf( x, loc = self.mean, scale = self.std ) ) + (x>=0)*(1-self.non_zero_proba)

    def dataType( self ):
        return 'real valued'




class zeroInflatedGamma( zeroInflatedDistributions ):
    
    def __init__(self, p, a, loc, scale ):
        zeroInflatedDistributions.__init__( self, p )
        if a <= 0:
            raise TypeError('The mean should be in (0,infty)' )
        self.gamma_parameter = a
        self.loc = loc
        self.scale = scale
        
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.gamma.rvs( self.gamma_parameter, loc = self.loc, scale = self.scale )
    
    def generate_nonzero_random_variable( self ):
        return stats.gamma.rvs(  self.gamma_parameter, loc = self.loc, scale = self.scale )    
    
    def mass_function( self, x ):
        return (x==0) * ( 1 - self.non_zero_proba ) + self.non_zero_proba * stats.gamma.pdf( x, self.gamma_parameter, loc = self.loc, scale = self.scale )
    
    def mass_function_withoutZeroInflation( self, x ):
        return self.non_zero_proba * stats.gamma.pdf( x , self.gamma_parameter, loc = self.loc, scale = self.scale )
    
    def cdf( self, x ):
        return self.non_zero_proba * stats.gamma.cdf( x, self.gamma_parameter, loc = self.loc, scale = self.scale ) + (x>=0)*(1-self.non_zero_proba)
    
    def dataType( self ):
        return 'real valued'





class zeroInflatedPareto( zeroInflatedDistributions ):
    
    def __init__(self, p, pareto_parameter, loc = 0, scale = 1 ):
        zeroInflatedDistributions.__init__( self, p )
        if pareto_parameter <= 0:
            raise TypeError('The mean should be in (0,infty)' )
        self.pareto_parameter = pareto_parameter
        self.loc = loc
        self.scale = scale
        
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.pareto.rvs( self.pareto_parameter, loc = self.loc, scale = self.scale )
    
    def generate_nonzero_random_variable( self ):
        return stats.pareto.rvs(  self.pareto_parameter, loc = self.loc, scale = self.scale )    
    
    def mass_function( self, x ):
        return (x==0) * ( 1 - self.non_zero_proba ) + self.non_zero_proba * stats.pareto.pdf( x, self.pareto_parameter, loc = self.loc, scale = self.scale )
    
    def mass_function_withoutZeroInflation( self, x ):
        return self.non_zero_proba * stats.pareto.pdf( x , self.pareto_parameter, loc = self.loc, scale = self.scale )
    
    def cdf( self, x ):
        return self.non_zero_proba * stats.pareto.cdf( x, self.pareto_parameter, loc = self.loc, scale = self.scale ) + (x>=0)*(1-self.non_zero_proba)

    def dataType( self ):
        return 'real valued'




"""
class inflatedGeometric(  ):
    
    def __init__(self, p, a ):
        if p<0 or p>1:
            raise TypeError( 'parameter p should be in [0,1]' )
        if a>1 or a<=0:
            raise TypeError('The geometric parameter should be in (0,1]' )
        self.non_zero_proba = p
        self.geometric_parameter = a

        
    def generate_random_variable( self ):
        if stats.bernoulli.rvs( self.non_zero_proba ) == 0:
            return 0
        else:
            return stats.geom.rvs( self.geometric_parameter )
        #return lambda x : 0 if  stats.bernoulli.rvs( self.non_zero_proba ) == 1 else stats.geom.rvs( self.geometric_parameter )
    
    def generate_nonzero_random_variable( self ):
        return stats.geom.rvs( self.geometric_parameter )
    
    def mass_function( self, x ):
        if x ==0 :
            return 1 - self.non_zero_proba
        else:
            return  self.non_zero_proba * stats.geom.pmf( x , self.geometric_parameter )
        #return lambda x : self.non_zero_proba if x == 0 else (1-self.non_zero_proba) * stats.geom.pmf( x , self.non_zero_proba ) if isinstance(1,int) else TypeError( 'x should be an integer' ) 
"""
