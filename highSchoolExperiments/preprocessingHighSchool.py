#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:46:06 2020

@author: mdreveto
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import community
import json
import os
import pandas as pd
import glob
from sklearn.cluster import SpectralClustering
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statistics
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


##High school contact and friendship networks


def preprocess_high_school_dataset_temporal_edges( groups_considered = [ ['2BIO1'], ['MP'] ] , year = 2013 ):
    path = 'datasets/'
    if (year == 2013):    
        path = path + 'High_school_contact_and_friendship_networks'
        file = 'High-School_data_2013.csv'
        file_metadata = 'metadata_2013.txt'
        sep = ' '
    elif year == 2012:
        path = path + 'High_school_dynamic_contact_networks'
        file = 'thiers_2012.csv'
        sep = '\t'
        file_metadata = 'metadata_2012.txt'
    elif year == 2011:
        path = path + 'High_school_dynamic_contact_networks'
        file = 'thiers_2011.csv'
        sep = '\t'
        file_metadata = 'metadata_2011.txt'

    os.chdir(path)
    files = glob.glob('*.csv')
    School = pd.read_csv(file,sep=sep, header=None)
    School.columns = ['Time','Node_1','Node_2','Class_1','Class_2']
    
    metadata = pd.read_csv(file_metadata,sep='\t', header=None)
        
    list_nodes = [*np.unique(np.hstack((np.unique(School.iloc[:,1]),np.unique(School.iloc[:,2]))))]
    nb_nodes = len(list_nodes)
    dict_nodes = dict( zip( list_nodes,[*range(nb_nodes)]) )
    
    School['Node_1'] = School['Node_1'].map(dict_nodes)
    School['Node_2'] = School['Node_2'].map(dict_nodes)    

    metadata.columns = ['Node','Class','Sex']
    metadata['Node'] = metadata['Node'].map(dict_nodes)
    metadata = metadata.dropna()
    
    print(pd.unique(School[ ['Class_1','Class_2'] ].values.ravel('K')))
    del files, list_nodes    
    
    K = len( groups_considered )
    groups_merged = []
    for group_cluster in groups_considered:
            groups_merged += group_cluster
    
    nodes_kept_list = [ ]
    for index, row in metadata.iterrows():
        if ( row['Class'] in groups_merged ):
            nodes_kept_list .append( row['Node'] )
    nodes_kept = pd.DataFrame( { 'Node' : nodes_kept_list } )
    nb_remaining_nodes = len( nodes_kept )
        
    School_truncated = School.merge( nodes_kept, left_on="Node_1", right_on="Node").merge( nodes_kept, left_on="Node_2", right_on="Node" )[ School.columns ]
    metadata_truncated = metadata.merge( nodes_kept, left_on="Node", right_on="Node")[ metadata.columns ]
    
    node_indexing = dict( )
    count = 0
    for index in range( nb_remaining_nodes ):
        node_indexing[ int( nodes_kept.iloc[index] ) ] = count
        count +=1
    
    labels_true = np.zeros( nb_remaining_nodes, dtype = 'int8' )
    labels = dict( )
    for k in range( K ):
        for group in groups_considered[ k ]:
            labels[ group ] = k + 1
    
    for index, row  in metadata_truncated.iterrows():
        labels_true[ node_indexing[ int( row['Node'] ) ] ] = labels[ row['Class'] ]
        
    sex = np.zeros( nb_remaining_nodes, dtype = 'int8' )
    for index, row  in metadata_truncated.iterrows():
        if row['Sex'] == 'F':
            sex[ node_indexing[ int( row['Node'] ) ] ] = 1
        else:
            sex[ node_indexing[ int( row['Node'] ) ] ] = 2
    
    time_indexing = dict( )
    count = 0
    times = list( set( School.Time.values) ) 
    times.sort()
    for t in times:
        time_indexing[t] = count
        count += 1
    
    
    missing_index = [ 0 ]
    missing_times = [School.iloc[0,0]]
    for index in range( 1, len( times ) ):
        if not ( times[index-1] == times[index] or times[index-1] == times[index] - 20 ):
            missing_index.append( index )
            missing_times.append( times[index] )
    missing_index.append(index) #Add the last index
            
    days = dict()
    days['Monday'] = [  t for t in range( missing_index[0], missing_index[1] ) ]
    days['Tuesday'] = [  t for t in range( missing_index[1], missing_index[2] ) ]
    days['Wednesday'] = [  t for t in range( missing_index[2], missing_index[3] ) ]
    days['Thursday'] = [  t for t in range( missing_index[3], missing_index[4] ) ]
    days['Friday'] = [  t for t in range( missing_index[4], missing_index[5]+1 ) ]
    
    temporal_edges = dict()
    for t in times:
        temporal_edges[ time_indexing[t ] ] = []
    
    for index, row in School_truncated.iterrows():        
        if(row['Node_1'] in nodes_kept_list and row['Node_2'] in nodes_kept_list ) :
            edge = ( node_indexing[ row['Node_1'] ], node_indexing[ row['Node_2'] ] ) 
            edge = tuple( np.sort(edge) )
            temporal_edges[ time_indexing[ row['Time'] ] ].append( edge)
        else:
            print('not in list')
    
    return labels_true, temporal_edges, days, node_indexing, sex



