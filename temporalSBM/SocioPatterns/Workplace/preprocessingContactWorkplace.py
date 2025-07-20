#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:44:54 2022

@author: dreveton
"""

import pandas as pd
import glob
import numpy as np

def preprocess_contactWorkplace_dataset_temporal_edges( groups_considered = 'all' , 
                                                  year = 2013,
                                                  discretize = True, keepNonInteractingPeriods = True ):
    
    """
    Pour 2013 les groupes sizes sont:
        'DISQ': 15
        'DSE': 34
        'SFLE': 4
        'DMCT': 26
        'SRH': 13
    Pour 2015 les groupes sizes sont (après avoir enlevé les noeuds qui n'interagissent jamais'):
        ['DCAR']: 13, 
        ['DG']: 2, 
        ['DISQ']: 18,
        ['DMCT']: 31,
        ['DMI']: 57,
        ['DSE']: 32,
        ['DST']: 23,
        ['SCOM']: 7,
        ['SDOC']: 4,
        ['SFLE']: 14,
        ['SRH']: 9,
        ['SSI']: 7 
    """
    
    if year == 2013:
        file_interactions = '2013_tij_InVS.dat'
        file_metadata = '2013_Department_list.txt'
        if groups_considered == 'all':
            groups_considered = [ ['DISQ'], ['DSE'], ['SFLE'], ['DMCT'], ['SRH'] ]
    elif year == 2015:
        file_interactions = '2015_tij_InVS.dat'
        file_metadata = '2015_Department_list.txt'
        if groups_considered == 'all':
            groups_considered = [['DCAR'], ['DG'], ['DISQ'],['DMCT'],['DMI'],['DSE'],['DST'],
                                 ['SCOM'],['SDOC'],['SFLE'],['SRH'],['SSI'] ]
    else:
        raise TypeError( 'The year should be 2013 or 2015' )
    

    interactions = pd.read_csv( file_interactions,sep = ' ', header=None)
    interactions.columns = [ 'Time','Node_1','Node_2' ]
    
    metadata = pd.read_csv( file_metadata,sep='\t', header=None)
    metadata.columns = [ 'Node','Community' ]    

    """
    if groups_considered == 'all':
        groups_merged = np.unique( metadata.iloc[:,1] )
        groups_considered = [ ]
        for group in groups_merged:
            groups_considered += [ [ group ] ]
    """
    
    #Now we are going to extract only the nodes that are in the groups we want
    K = len( groups_considered )
    groups_merged = []
    for group_cluster in groups_considered:
        groups_merged += group_cluster
    
    nodes_kept_list = [ ]
    for index, row in metadata.iterrows():
        if ( row[ 'Community' ] in groups_merged ):
            nodes_kept_list .append( row['Node'] )
    nodes_kept = pd.DataFrame( { 'Node' : nodes_kept_list } )
    nb_remaining_nodes = len( nodes_kept )
        
    interactions_truncated = interactions.merge( nodes_kept, left_on = "Node_1", right_on = "Node" ).merge( nodes_kept, left_on = "Node_2", right_on = "Node" )[ interactions.columns ]
    metadata_truncated = metadata.merge( nodes_kept, left_on = "Node", right_on = "Node" )[ metadata.columns ]
    
    #We keep only the nodes labels which are interacting
    interacting_nodes = [ *np.unique(np.hstack( ( np.unique(interactions_truncated.iloc[:,1]),np.unique(interactions_truncated.iloc[:,2] ) ) ) ) ]
    
    dict_nodes = dict( )
    nb_interacting_nodes = 0
    for i in range( len(metadata_truncated ) ):
        if metadata_truncated[ 'Node' ][ i ] in interacting_nodes:
            dict_nodes[ metadata_truncated[ 'Node' ][ i ] ] = nb_interacting_nodes
            nb_interacting_nodes += 1
        
    if nb_interacting_nodes != nb_remaining_nodes:
        print( 'Some nodes have zero interactions, we will drop them' )
    
    #We rename the nodes from 0 to N-1
    interactions_truncated['Node_1'] = interactions_truncated['Node_1'].map(dict_nodes)
    interactions_truncated['Node_2'] = interactions_truncated['Node_2'].map(dict_nodes)    
    
    metadata_truncated['Node'] = metadata_truncated['Node'].map(dict_nodes)
    metadata_truncated = metadata_truncated.dropna( )
    
    
    
    labels_true = np.zeros( nb_interacting_nodes, dtype = 'int8' )
    labels = dict( )
    for k in range( K ):
        for group in groups_considered[ k ]:
            labels[ group ] = k + 1
    for index, row  in metadata_truncated.iterrows():
        labels_true[ int( row['Node'] ) ] = labels[ row['Community'] ]
    
    
    
    #days = [ 'Monday1', 'Tuesday2', 'Wednesday3', 'Thursday4', 'Friday5', 'Monday6', 'Tuesday7', 'Wednesday8' ]
        
    times = list( set( interactions_truncated.Time.values) ) 
    times.sort( )
    
    differences = [ times[t] - times[t-1] for t in range( 1, len(times) ) ]
    lastTimeOfTheDay = [ ]
    firstTimeOfTheDay = [ times[ 0 ] ]
    for t in range( len( differences) ):
        if differences[ t ] > 36000: 
            #This means that the time difference between two is larger than 36000s (10h): we are at two different days.
            lastTimeOfTheDay.append( times[ t ] )
            firstTimeOfTheDay.append( times[ t + 1 ] )
            
    lastTimeOfTheDay.append( times[-1] )
    lastTimeOfTheDay.sort( )
    firstTimeOfTheDay.sort( )
    
    days = dict()
    temporal_edges = dict()
    time_indexing = dict( )
    
    nb_days = len( firstTimeOfTheDay )
    for day in range( nb_days ):
        days[ day ] = [ t for t in range( firstTimeOfTheDay[ day ], lastTimeOfTheDay[ day ] + 1 , 20 ) ]
        if keepNonInteractingPeriods == False:
            days[ day ] = set( days[day]).intersection( set( times) )
            days[ day ] = list( days[ day ] )
            days[ day ].sort( )
            
        temporal_edges[ day ] = dict( )
        
        nb_timesteps = 0
        for t in days[ day ]:
            time_indexing[ t ] = nb_timesteps        
            temporal_edges[ day ][ nb_timesteps ] = [ ]
            nb_timesteps += 1
            
    
    for index, row in interactions_truncated.iterrows( ):        
        
        edge = ( row['Node_1'], row['Node_2'] )
        edge = tuple( np.sort( edge ) )
        interaction_time = row['Time']
        day = isInDay( interaction_time, firstTimeOfTheDay, lastTimeOfTheDay )
        temporal_edges[ day ][ time_indexing[ interaction_time ] ].append( edge )

        
    return temporal_edges, labels_true




def isInDay( timestep, firstTimeOfTheDay, lastTimeOfTheDay ):
    
    for day in range( len( firstTimeOfTheDay ) ):
        if timestep >= firstTimeOfTheDay[ day ] and timestep <= lastTimeOfTheDay[ day ]:
            return day
    
    raise TypeError( 'Cannot find the correct day in which the timestep belongs to')