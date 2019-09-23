# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:20:18 2015

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""

import numpy
import random
import sklearn.discriminant_analysis.LinearDiscriminantAnalysis as LDA
from sklearn.decomposition.pca import PCA
from sklearn.decomposition import NMF
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics


class SpaceMapper:
    def __init__(self):
        self.npc = 10
        self.scaler = None
        self.pca_transformer = None
        self.lda_transformer = None
        self.nmf_transformer = None
        self.traindata = None
        self.testdata = None
        self.valdata = None
        self.trainlabels = None
        self.trainaudiolabels = None
        self.vallabels = None
        self.valaudiolabels = None
        self.testlabels = None
        self.testaudiolabels = None
        self.pred_frame_labels = None
        self.pred_recording_labels = None
        self.recording_labels = None
        self.max_frame_accuracy = -1        
        self.max_recording_accuracy = -1
        self.pca_traindata = None
        self.lda_traindata = None
        self.nmf_traindata = None
        self.pca_testdata = None
        self.lda_testdata = None
        self.nmf_testdata = None
        self.pca_valdata = None
        self.lda_valdata = None
        self.nmf_valdata = None        


    def learn_train_space(self, npc=None, traindata=None, trainlabels=None):
        """ learn PCA, LDA, NMF space and transform train data
        """        
        # initialize train data        
        if traindata is not None:
            self.traindata = traindata
        if trainlabels is not None:
            self.trainlabels = trainlabels
        if npc is not None:
            self.npc = npc        

        # learn space and transform train data
        random.seed(254678)
        # standardize (samples)
        self.traindata = scale(self.traindata, axis=1)
        # then pca
        print "training with PCA transform..."
        self.pca_transformer = PCA(n_components=npc).fit(self.traindata)
        self.pca_traindata = self.pca_transformer.transform(self.traindata)
        # then lda
        print "training with LDA transform..."
        self.lda_transformer = LDA(n_components=npc).fit(self.traindata, self.trainlabels)
        self.lda_traindata = self.lda_transformer.transform(self.traindata)
        # then nmf
        print "training with NMF transform..."
        self.nmf_transformer = NMF(n_components=npc).fit(self.traindata-numpy.min(self.traindata))
        self.nmf_traindata = self.nmf_transformer.transform(self.traindata-numpy.min(self.traindata))


    def map_val_space(self, valdata=None, vallabels=None):
        """ map validation space
        """
        # initialize validation data    
        if valdata is not None:
            self.valdata = valdata
        if vallabels is not None:
            self.vallabels = vallabels
        
        # transform validation data        
        random.seed(3759137)
        self.valdata = scale(self.valdata, axis=1)
        print "transform val data..."
        self.pca_valdata = self.pca_transformer.transform(self.valdata)
        self.lda_valdata = self.lda_transformer.transform(self.valdata)
        self.nmf_valdata = self.nmf_transformer.transform(self.valdata-numpy.min(self.valdata))


    def map_test_space(self, testdata=None, testlabels=None):
        """ map test space
        """
        # initialize test data    
        if testdata is not None:
            self.testdata = testdata
        if testlabels is not None:
            self.testlabels = testlabels
        
        # transform test data
        random.seed(3759137)
        self.testdata = scale(self.testdata, axis=1)
        print "transform test data..."
        self.pca_testdata = self.pca_transformer.transform(self.testdata)
        self.lda_testdata = self.lda_transformer.transform(self.testdata)
        self.nmf_testdata = self.nmf_transformer.transform(self.testdata-numpy.min(self.testdata))


    def classify_frames_recordings(self, model, transform_name="", model_name=""):
        """ predictions per frame and per recording
        """
        # classification accuracy per frame  
        if transform_name == "":
            acc_frame, preds_frame = self.classify(model, self.traindata, self.trainlabels, self.testdata, self.testlabels)        
        elif transform_name == "PCA":
            acc_frame, preds_frame = self.classify(model, self.pca_traindata, self.trainlabels, self.pca_testdata, self.testlabels)
        elif transform_name == "LDA":
            acc_frame, preds_frame = self.classify(model, self.lda_traindata, self.trainlabels, self.lda_testdata, self.testlabels)
        elif transform_name == "NMF":
            acc_frame, preds_frame = self.classify(model, self.nmf_traindata, self.trainlabels, self.nmf_testdata, self.testlabels)
        
        # classification accuracy per recording by a vote count
        acc_vote, preds_vote = self.vote_count(preds_frame) 
        print model_name+" "+transform_name+" "+str(acc_frame)+" "+str(acc_vote)    

        # update highest accuracy and predictions per frame and per recording
        if acc_vote > self.max_recording_accuracy:
            self.pred_frame_labels = preds_frame
            self.pred_recording_labels = preds_vote
            self.max_frame_accuracy = acc_frame
            self.max_recording_accuracy = acc_vote

    
    def evalaluate_space(self, traindata=None, trainlabels=None, testdata=None, testlabels=None, audiolabels=None):
        """ evaluate space by classification
        """
        # initialize data (features, labels, audiolabels)        
        if self.traindata is None:
            self.learn_train_space(traindata=traindata, trainlabels=trainlabels)
        if self.testdata is None:
            self.map_test_space(testdata=testdata, testlabels=testlabels)
        if self.testaudiolabels is None:
            self.testaudiolabels = audiolabels
        
        # initialize classifiers        
        modelKNN = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        modelLDA = LDA()
        modelSVM = svm.SVC(kernel='rbf', gamma=0.1)    
        transforms = ["", "PCA", "LDA", "NMF"]
        
        # predict labels per frame and per recording 
        print "classify with KNN..."
        for transform in transforms:
            self.classify_frames_recordings(modelKNN, transform_name=transform, model_name="KNN")
        print "classify with LDA..."
        for transform in transforms:
            self.classify_frames_recordings(modelLDA, transform_name=transform, model_name="LDA")
        print "classify with SVM..."
        for transform in transforms:
            self.classify_frames_recordings(modelSVM, transform_name=transform, model_name="SVM")


    def classify(self, model, traindata, trainlabels, testdata, testlabels):
        """ train classifier and return predictions and accuracy on test set
        """        
        model.fit(traindata, trainlabels)
        predlabels = model.predict(testdata)
        accuracy = metrics.accuracy_score(testlabels, predlabels)
        return accuracy, predlabels

    
    def vote_count(self, preds_frame):
        """ return predictions per recording by a vote count of predictions per frame
        """
        # initialize
        uniq_audiolabels = numpy.unique(self.testaudiolabels)
        preds_vote = []
        true_labels = []
        
        # get prediction vote count and true label for each recording
        for audio_label in uniq_audiolabels:
            inds = numpy.where(self.testaudiolabels==audio_label)[0]
            preds, counts = numpy.unique(preds_frame[inds], return_counts=True)
            preds_vote.append(preds[numpy.argmax(counts)])
            true_labels.append(numpy.unique(self.testlabels[inds]))
        
        # return accuracy and predictions per recording
        preds_vote = numpy.array(preds_vote)
        true_labels = numpy.array(true_labels)
        accuracy = metrics.accuracy_score(true_labels, preds_vote)
        self.recording_labels = true_labels  # todo: this should only be assigned once
        return accuracy, preds_vote


if __name__ == '__main__':
    SpaceMapper()
