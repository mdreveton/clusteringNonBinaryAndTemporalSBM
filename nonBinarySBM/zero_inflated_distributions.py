#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:29:23 2021

@author: mdreveto
"""

class zeroInflatedDistribution( ):
    
    def __init__( self, p ):
        if p<0 or p>1:
            raise TypeError( 'parameter p should be in [0,1]' )
        self.non_zero_proba = p
    