# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:36:44 2016

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""
""" split in train, val, test sets and load features and labels """

import numpy
import pandas
import pickle
import os
import sys
sys.path.append('util')
import process_frames


def subset_classLabels(class_labels, N=30, seed=None):
    """ select at random N instances for each class and return indices
    """
    if seed is not None:
        numpy.random.seed(seed)
    subset_inds = []
    unique_class, counts = numpy.unique(class_labels, return_counts=True)
    for i_class in unique_class:
        class_inds = numpy.where(class_labels==i_class)[0]
        if len(class_inds)>=N:
            subset_inds.append(numpy.random.choice(class_inds, int(N), replace=False))
    subset_inds = numpy.concatenate(subset_inds, axis=0)
    return subset_inds


def load_and_post_process_frames(dfrhy, dfmel, dftimb, dfharm, inds=None):
    """ load rhythm, melody, timbre, harmony features and post process
    """
	
    print(inds)
    # load features and do some post processing
    frames_rhy = process_frames.load_frames(df=dfrhy.iloc[inds, :], K=2, testclass=testclass)
    frames_mel = process_frames.load_frames(df=dfmel.iloc[inds, :], nmfpb=True, testclass=testclass)
    frames_timb = process_frames.load_frames(df=dftimb.iloc[inds, :], testclass=testclass, deltamfcc=True, avelocalframes=True)
    frames_harm = process_frames.load_frames(df=dfharm.iloc[inds, :], testclass=testclass, alignchroma=True, avelocalframes=True)    

    # load labels
    frameclasslabels = numpy.asarray(frames_rhy[testclass].get_values(), dtype='str')
    frameaudiolabels = numpy.asarray(frames_rhy["Audio"].get_values(), dtype='str')
    
    # concatenate rhythm, melody, timbre, harmony features
    framefeat = numpy.concatenate((frames_rhy,frames_mel,frames_timb,frames_harm), axis=1)
    return framefeat, frameclasslabels, frameaudiolabels


if __name__ == '__main__':
    # spit in train, validation and test sets
    df = pandas.read_csv("data/Metadata.csv", engine="c", header=0)
    testclass = "Country"
    N = 70
    testclass_labels = numpy.array(df[testclass].get_values(), dtype='str')
    randomseed = 1234
    trainvalinds = subset_classLabels(testclass_labels, N=round(0.8*N), seed=randomseed)
    testinds = numpy.array(list(set(range(len(testclass_labels)))-set(trainvalinds)))
    valinds = trainvalinds[subset_classLabels(testclass_labels[trainvalinds], N=round(0.2*N), seed=randomseed)]
    traininds = numpy.array(list(set(trainvalinds)-set(valinds)))
    
    # load path to csv features
    dfrhy = process_frames.load_data(audiolist="data/audiolist.txt", metadatacsv="data/Metadata.csv", csvlist="data/csvlist_rhythm.txt", sepcsv='\t', headermeta=0)
    dfmel = process_frames.load_data(audiolist="data/audiolist.txt", metadatacsv="data/Metadata.csv", csvlist="data/csvlist_melody.txt", sepcsv='\t', headermeta=0)
    dftimb = process_frames.load_data(audiolist="data/audiolist.txt", metadatacsv="data/Metadata.csv", csvlist="data/csvlist_timbre.txt", sepcsv='\t', headermeta=0)
    dfharm = process_frames.load_data(audiolist="data/audiolist.txt", metadatacsv="data/Metadata.csv", csvlist="data/csvlist_harmony.txt", sepcsv='\t', headermeta=0)
    
    train_set = load_and_post_process_frames(dfrhy, dfmel, dftimb, dfharm, inds=traininds)
    val_set = load_and_post_process_frames(dfrhy, dfmel, dftimb, dfharm, inds=valinds)
    test_set = load_and_post_process_frames(dfrhy, dfmel, dftimb, dfharm, inds=testinds)
    
    write_output = False
    if write_output:
        pickle.dump([train_set, val_set, test_set], open(os.path.join('data', 'dataset.pickle'), 'wb'))