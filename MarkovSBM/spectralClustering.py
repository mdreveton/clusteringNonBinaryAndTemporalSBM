#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:14:06 2022

@author: maximilien
"""
import numpy as np
from sklearn.cluster import SpectralClustering


def staticSpectralClustering( adjacencyMatrix, K = 2, assign_labels = 'kmeans' ):
    
    sc = SpectralClustering( n_clusters = K, affinity = 'precomputed', assign_labels = assign_labels )
    labels_pred_spec = sc.fit_predict( adjacencyMatrix ) + np.ones( adjacencyMatrix.shape[0] )
    
    return labels_pred_spec.astype( int )



