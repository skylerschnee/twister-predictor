#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math
import os
from datetime import timedelta, datetime

def get_tornado_data():
    '''
    get_tornado_data pulls in the original, unclean datafram as it comes from 
    the source; https://www.kaggle.com/datasets/danbraswell/us-tornado-dataset-1950-2021
    '''
    # assign filename
    filename = 'tornado_df.csv'
    # assign csv as df
    df = pd.read_csv(filename, index_col=0)    
    return df

def wrangle_tornado_data():
    '''
    wrangle_tornado_data pulls in df using above function, converts index to datetime,
    renames cols, drops redundant cols, creats a new col, and returns the wrangled df.
    '''
    # pull original df from above funtion
    df = get_tornado_data()
    # convert to datetime and set as index
    df.index = pd.to_datetime(df.date).sort_index()
    # drop old datetime cols
    df = df.drop(columns = ['mo', 'dy', 'date'])
    # rename cols to promote readability
    df = df.rename(columns = {"st": "state", "mag": "ef", "inj": "injuries", 
                     "fat": "fatalities", "slat": "s_lat", "slon": "s_lon",
                     "elat": "e_lat", "elon": "e_lon", "len": "length", 
                     "wid": "width"})
    # creat new col with sum of injuries and fatalities
    df['casualties'] = df.injuries + df.fatalities
    # add new col calculating people effected physically by tornadoes per mile of tornadic track
    df['cas_per_mile'] = df.apply(
    lambda x: (x['injuries'] + x['fatalities']) / x['length'] if (
        x['length'] != 0) else 0, axis=1)
    # round that col to promote readability
    df['cas_per_mile'] = round(df['cas_per_mile'],2)
    # split data
    train, test = split_data(df, train_size=0.8)
    # return wrangled dfs
    return df, train, test

def get_df_by_month(df):
    '''
    get_df_by_month takes in wrangled tornado df, resamples by month and aggregates
    columns with the proper method for timeseries data
    '''
    #create alternate df by resampling by the year
    df_by_month = df.resample('M').agg({
    'injuries': 'sum',
    'fatalities': 'sum',
    'length': 'mean',
    'width': 'mean',
    'casualties': 'sum',
    'cas_per_mile': 'mean'
    })
    return df_by_month

def get_df_by_year(df):
    '''
    get_df_by_year takes in wrangled tornado df, resamples by year and aggregates
    columns with the proper method for that data
    '''
    #create alternate df by resampling by the year
    df_by_year = df.resample('Y').agg({
    'injuries': 'sum',
    'fatalities': 'sum',
    'length': 'mean',
    'width': 'mean',
    'casualties': 'sum',
    'cas_per_mile': 'mean'
    })
    return df_by_year


def get_df_by_decade(df):
    '''
    get_df_by_year takes in wrangled tornado df, resamples by year and aggregates
    columns with the proper method for that data
    '''
    #create alternate df by resampling by the year
    df_by_year = df.resample('10Y').agg({
    'injuries': 'sum',
    'fatalities': 'sum',
    'length': 'mean',
    'width': 'mean',
    'casualties': 'sum',
    'cas_per_mile': 'mean'
    })
    return df_by_year
    
    
def split_data(df, train_size):
    '''
    Splits a time-series dataframe into training and testing sets based on a given train size.
    :param df: time-series dataframe to be split
    :param train_size: proportion of data to use for training (default: 0.8)
    :return: tuple of (training set, test set)
    '''
    n = df.shape[0]
    test_start_index = round(train_size * n)

    train = df[:test_start_index] # everything up (not including) to the test_start_index
    test = df[test_start_index:] # everything from the test_start_index to the end
    
    return train, test 


