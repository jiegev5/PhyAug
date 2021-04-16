#!/usr/bin/env python
# coding: utf-8

# # import libs

# In[13]:


import pandas as pd
import numpy as np


# how many classes per side, for example: if 25 classes, then each dim is 5 for x and y
Class_dim = 5

def read_csv_to_np(filename):
    ##read csv file
    # filename = 'model_ch_src_1000_rec_10_data_1.csv'
    raw_data = pd.read_csv(filename)
    
    # split into features and classes
    df1 = raw_data.iloc[:,:10]
    df2 = raw_data.iloc[:,10:]
    
    # process df2, convert loc into classes
    # src_loc = ((np.floor(df2) - 1).astype(int)).to_numpy()
    src_loc = np.floor((df2 - 1)/(100/Class_dim)).astype(int).to_numpy()
    # get the receiver loc
    src_linear_loc = [np.ravel_multi_index(item, dims=(Class_dim,Class_dim), order='F') for item in src_loc]
    
    
    # sensor loc: fixed
    RecLoc = np.array([[4.3756,36.9532],
                       [78.4427,27.4769],
                       [33.3973,34.3930],
                       [81.9302, 9.6091],
                       [18.1883,45.7044],
                       [67.9357,46.0117],
                       [87.6844, 3.8745],
                       [75.9584,64.0731],
                       [23.7329, 6.8864],
                       [36.5354,17.7504]])
    
    # the RecLoc is MatLab based, which starts from 1, wheras python starts from 0
    RecLoc = (np.floor(RecLoc) - 1)
    RecLoc = RecLoc.astype(int)
    #  linear index equivalents from 2D -> 1D: sub2ind on matlab
    Recloc_linear_index = [np.ravel_multi_index(item, dims=(100,100), order='F') for item in RecLoc]
    
    # create 1000 100x100 images
    output = np.zeros((1000,10000))
    
    # map travel time from each sensor to the map
    feature = df1.to_numpy()
    for i in range(10):
        output[:,Recloc_linear_index[i]] = output[:,Recloc_linear_index[i]] + feature[:,i] 
    
    
    src_loc = np.asarray(src_linear_loc)
    src_loc = src_loc.reshape(1000,1)
    
    
    
    final_out = np.concatenate((output,src_loc),axis = 1)
    
    return final_out
    
    
    
# im1_2d = im1.reshape(100,100,order = 'F')
# np.savetxt("model_ch_src_1000_rec_10_data_1_converted.csv", final_out, delimiter=",")
    
    
    
    
    
