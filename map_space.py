# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 17:20:20 2016

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""

import numpy
import pickle
import os
import sys
sys.path.append('util')
import space_mapper as sm


def find_optimal_components(train_set, val_set, min_npc=5, max_npc=None):
    """ find optimal number of components for PCA, LDA, NMF decomposition"""
    traindata, trainlabels, trainaudiolabels = train_set
    valdata, vallabels, valaudiolabels = val_set
    if max_npc is None:
        max_npc = len(numpy.unique(vallabels))

    ssm = sm.SpaceMapper()    
    ssm.testaudiolabels = valaudiolabels
    accuracy = []
    for npc in range(min_npc, max_npc):
        ssm.learn_train_space(npc=npc, traindata=traindata, trainlabels=trainlabels)
        ssm.map_test_space(testdata=valdata, testlabels=vallabels)
        ssm.evaluate_space()
        accuracy.append(ssm.max_recording_accuracy)
    best_npc = numpy.argmax(numpy.asarray(accuracy)) + min_npc
    return best_npc


def learn_and_map_space(train_set, val_set, test_set, best_npc=None):
    """ train PCA, LDA, NMF transformers, map train, val, and test sets, and 
        evaluate transformed space by classification
    """
    traindata, trainlabels, trainaudiolabels = train_set
    valdata, vallabels, valaudiolabels = val_set
    testdata, testlabels, testaudiolabels = test_set    
    
    # learn space and map on train set
    ssm = sm.SpaceMapper()    
    ssm.trainaudiolabels = trainaudiolabels
    ssm.learn_train_space(npc=best_npc, traindata=traindata, trainlabels=trainlabels)
    
    # map space on validation set
    ssm.valaudiolabels = valaudiolabels
    ssm.map_val_space(valdata=valdata, vallabels=vallabels)
    
    # map space on test set
    ssm.testaudiolabels = testaudiolabels
    ssm.map_test_space(testdata=testdata, testlabels=testlabels)
    
    # evaluate space by classification
    ssm.evaluate_space()
    return ssm


def get_data_predictions(ssm=None):
    """ return lda-transformed data (best classification accuracy) and
        true/predicted labels
    """
    ldadata = numpy.concatenate((ssm.lda_traindata, ssm.lda_valdata, ssm.lda_testdata), axis=0)
    audiolabels = numpy.concatenate((ssm.trainaudiolabels, ssm.valaudiolabels, ssm.testaudiolabels), axis=0)
    labels = numpy.concatenate((ssm.trainlabels, ssm.vallabels, ssm.testlabels), axis=0)
    data = [ldadata, labels, audiolabels]
    predictions = [ssm.recording_labels, ssm.pred_recording_labels]
    return data, predictions


if __name__ == '__main__':
    # load dataset
    train_set, val_set, test_set = pickle.load(open(os.path.join('data', 'dataset.pickle'), 'rb'))
    
    #best_npc = find_optimal_components(train_set, val_set)
    best_npc = 30  # optimal performance at 30 principal components

    ssm = learn_and_map_space(train_set, val_set, test_set, best_npc=best_npc)
    data, predictions = get_data_predictions(ssm) #Change ssm = None to ssm_Group_4
    
    # write pickle file with transformed train, validation and test set
    write_output = True
    if write_output:
        pickle.dump([data, predictions], open(os.path.join('data', 'feature_space.pickle'), 'wb'))
