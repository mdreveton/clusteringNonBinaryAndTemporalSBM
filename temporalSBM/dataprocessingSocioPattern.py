#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:36:55 2022

@author: maximilien
"""

import pandas as pd
import glob
import numpy as np

import os
working_directory_path = os.getcwd() # Check current directory's path


def getAllDatasetNames( ):
    dataset_names = [ 'high school 2011', 'high school 2012', 'high school 2013',
                'workplace 2013', 'workplace 2015',
                'primary school' ]
    return dataset_names


def getDatasetPath( dataset ):
    
    working_directory_path = os.getcwd() # Obtain current directory's path
    path = working_directory_path + '/SocioPatterns'
        
    if dataset == 'workplace 2013':
        path = path + '/Workplace/'
        file_interactions = path + '2013_tij_InVS.dat'
        file_metadata = path + '2013_Department_list.txt'
        sep = ' '
    
    elif dataset == 'workplace 2015':
        path = path + '/Workplace/'
        file_interactions = path + '2015_tij_InVS.dat'
        file_metadata = path + '2015_Department_list.txt'
        sep = ' '
    
    elif dataset == 'high school 2011':
        path = path = path + '/High_School/'
        file_interactions = path + '2011_thiers.csv'
        file_metadata = path + '2011_metadata.txt'
        sep = '\t'
    
    elif dataset == 'high school 2012':
        path = path + '/High_School/'
        file_interactions = path + '2012_thiers.csv'
        file_metadata = path + '2012_metadata.txt'
        sep = '\t'
    
    elif dataset == 'high school 2013':
        path = path + '/High_School/'
        file_interactions = path + '2013_thiers.csv'
        file_metadata = path + '2013_metadata.txt'
        sep = ' '
    
    elif dataset == 'primary school':
        path = path + '/Primary_School/'
        file_interactions = path + 'primaryschool.csv'
        file_metadata = path + 'primaryschool_metadata.txt'
        sep = '\t'
        
    else:
        raise TypeError( 'This dataset is not implemented' )

    return file_interactions, file_metadata, sep


def dataProcessingSocioPatterns( dataset = 'high school 2011',
                                communities = 'all', 
                                keepNonInteractingPeriods = True,
                                sex = False ):
    
    dataset = dataset.lower( )
    file_interactions, file_metadata, sep = getDatasetPath( dataset ) 

    interactions = pd.read_csv( file_interactions, sep = sep, header = None )
    if dataset in [ 'high school 2011', 'high school 2012', 'high school 2013', 'primary school']:
        interactions = interactions.drop( columns = [3,4] )
    
    interactions.columns = [ 'Time', 'Node_1', 'Node_2' ]
    
    metadata = pd.read_csv( file_metadata, sep = '\t', header=None )
    if dataset in [ 'high school 2011', 'high school 2012', 'high school 2013', 'primary school' ]:
        metadata.columns = [ 'Node','Community', 'Sex' ]
    else:
        metadata.columns = [ 'Node','Community' ]

    if communities == 'all':
        communities_merged = np.unique( metadata.iloc[:,1] )
        communities = [ ]
        for group in communities_merged:
            communities += [ [ group ] ]
    else:
        communities_merged = []
        for community in communities:
            communities_merged += community
    

    #Now we are going to extract only the nodes that are in the groups we want
    K = len( communities )
    nodes_kept_list = [ ]
    for index, row in metadata.iterrows():
        if ( row[ 'Community' ] in communities_merged ):
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
        print( nb_remaining_nodes - nb_interacting_nodes, ' nodes have zero interactions, we will drop them' )
    
    #We rename the nodes from 0 to N-1
    interactions_truncated['Node_1'] = interactions_truncated['Node_1'].map(dict_nodes)
    interactions_truncated['Node_2'] = interactions_truncated['Node_2'].map(dict_nodes)    
    
    metadata_truncated['Node'] = metadata_truncated['Node'].map(dict_nodes)
    metadata_truncated = metadata_truncated.dropna( )
    
    
    
    labels_true = np.zeros( nb_interacting_nodes, dtype = 'int8' )
    labels = dict( )
    for k in range( K ):
        for group in communities[ k ]:
            labels[ group ] = k + 1
    for index, row  in metadata_truncated.iterrows():
    	if sex == False:
        	labels_true[ int( row['Node'] ) ] = labels[ row['Community'] ] 
    	else:
            if row['Sex'] == 'F':
                labels_true[ int( row['Node'] ) ] = 1
            elif row['Sex'] == 'M':
                labels_true[ int( row['Node'] ) ] = 2
            else:
                labels_true[ int( row['Node'] ) ] = 3
    #days = [ 'Monday1', 'Tuesday2', 'Wednesday3', 'Thursday4', 'Friday5', 'Monday6', 'Tuesday7', 'Wednesday8' ]
        
    times = list( set( interactions_truncated.Time.values) ) 
    times.sort( )
    
    differences = [ times[t] - times[t-1] for t in range( 1, len(times) ) ]
    lastTimeOfTheDay = [ ]
    firstTimeOfTheDay = [ times[ 0 ] ]
    for t in range( len( differences) ):
        if differences[ t ] > 36000: 
            #This means that the time difference between two is larger than 36000s (10h): we have two different days.
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
    
    
    nb_edges_droped = 0
    for index, row in interactions_truncated.iterrows( ):        
        
        edge = ( row['Node_1'], row['Node_2'] )
        edge = tuple( np.sort( edge ) )
        interaction_time = row['Time']
        day = isInDay( interaction_time, firstTimeOfTheDay, lastTimeOfTheDay )
        try:
            temporal_edges[ day ][ time_indexing[ interaction_time ] ].append( edge )
        except KeyError:
            #print("Oops!  Some timesteps have been droped")
            nb_edges_droped += 1
            continue
    
    if nb_edges_droped!=0:
        print( nb_edges_droped, ' edges have been dropped' )
        
    return temporal_edges, labels_true, labels




def isInDay( timestep, firstTimeOfTheDay, lastTimeOfTheDay ):
    
    for day in range( len( firstTimeOfTheDay ) ):
        if timestep >= firstTimeOfTheDay[ day ] and timestep <= lastTimeOfTheDay[ day ]:
            return day
    
    raise TypeError( 'Cannot find the correct day in which the timestep belongs to')
    
    
    
    
def datasetsProperties( dataset = 'high school 2011' ):
    properties = dict( )
    dataset = dataset.lower()
    
    if dataset == 'workplace contact 2013':
        properties[ 'DISQ' ] = 15
        properties[ 'DSE' ] = 34
        properties[ 'SFLE' ] = 4
        properties[ 'DMCT' ] = 26
        properties[ 'SRH' ] = 13
        
    elif dataset == 'workplace contact 2015':
        properties['DCAR'] = 13
        properties['DG'] =  2
        properties['DISQ'] = 18
        properties[ 'DMCT' ] = 31
        properties[ 'DMI' ] = 57
        properties[ 'DSE' ] = 32
        properties['DST'] = 23
        properties[ 'SCOM' ] = 7
        properties[ 'SDOC' ] = 4
        properties[ 'SFLE' ] = 14,
        properties[ 'SRH' ] = 9,
        properties[ 'SSI' ] = 7 
        
    elif dataset == 'high school 2011':
        properties['PC'] = 31
        properties['PC*'] =  45
        properties['PSI*'] = 42
        properties['teacher'] = 4
        
    elif dataset == 'high school 2012':
        properties['MP*1'] = 31
        properties['MP*2'] =  35
        properties['PC'] = 38
        properties['PC*'] = 35
        properties['PSI*'] = 41
    
    elif dataset == 'high school 2013':
        properties['2BIO1'] = 36
        properties['2BIO2'] = 34
        properties['2BIO3'] = 40
        properties['MP'] = 33
        properties['MP*1'] = 29
        properties['MP*2'] =  38
        properties['PC'] = 44
        properties['PC*'] = 39
        properties['PSI*'] = 34
    
    elif dataset == 'primary school':
        properties['1A'] = 23
        properties['1B'] = 25
        properties['2A'] = 23
        properties['2B'] = 26
        properties['3A'] = 23
        properties['3B'] = 22
        properties['4A'] = 21
        properties['4B'] = 23
        properties['5A'] = 22
        properties['5B'] = 24
        properties['Teachers'] = 10
        
    
    return properties



def getAllCommunities( dataset ):
    
    if dataset == 'workplace contact 2013':
            communities = [ ['DISQ'], ['DSE'], ['SFLE'], ['DMCT'], ['SRH'] ]
    
    elif dataset == 'workplace contact 2015':
            communities = [ ['DCAR'], ['DG'], ['DISQ'],['DMCT'],['DMI'],['DSE'],['DST'],
                                 ['SCOM'],['SDOC'],['SFLE'],['SRH'],['SSI'] ]
    
    else:
        raise TypeError( 'This dataset is not implemented' )
        
    return communities

