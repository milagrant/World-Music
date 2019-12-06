# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:15:09 2016

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""
""" print results, plot confusion matrix and outliers """

import numpy
import pickle
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.covariance import MinCovDet
from scipy import stats
from collections import Counter
import os


def plot_conf_matrix(conf_matrix, labels=None, figurename=None):
    """ plot the confusion matrix with country labels
    """
    labels[labels=='United States of America'] = 'United States Amer.'
    plt.imshow(conf_matrix, cmap="gray")
    plt.xticks(range(len(labels)), labels, rotation='vertical', fontsize=7)
    plt.yticks(range(len(labels)), labels, fontsize=7)
    if figurename is not None:
        plt.savefig(figurename, bbox_inches='tight')


def average_frames(features, classlabels, audiolabels):
    """ average frames and group labels for each recording 
    """
    u, ind = numpy.unique(audiolabels, return_index=True)
    uniqsorted = u[numpy.argsort(ind)]
    newfeatures = []
    newclasslabels = []
    newaudiolabels = []
    for aulabel in uniqsorted:
        inds = numpy.where(audiolabels == aulabel)[0]
        newfeatures.append(numpy.mean(features[inds, :], axis=0))
        newclasslabels.append(classlabels[inds[0]])
        newaudiolabels.append(aulabel)
    newfeatures = numpy.asarray(newfeatures)
    newaudiolabels = numpy.asarray(newaudiolabels)
    newclasslabels = numpy.asarray(newclasslabels)
    return newfeatures, newclasslabels, newaudiolabels


def get_outliers(X, chi2thr=0.975, plot=False, figurename=None):
    """ detect outliers by Mahalanobis distance
    """
    robust_cov = MinCovDet(random_state=100).fit(X)
    MD = robust_cov.mahalanobis(X)
    n_samples = len(MD)
    chi2 = stats.chi2
    degrees_of_freedom = X.shape[1]
    threshold = chi2.ppf(chi2thr, degrees_of_freedom)
    y_pred = MD>threshold
    outlierpercent = sum(y_pred)/float(n_samples)
    return outlierpercent, y_pred, MD


def plot_outliers(MD, y_pred, figurename=None):
    """ plot Mahalanibis distance for each recording
    """
    n_samples = len(MD)
    colors = numpy.repeat("black", n_samples, axis=0)
    colors[y_pred] = "red"
    markers = numpy.repeat(".", n_samples, axis=0)
    markers[y_pred] = "+"
    plt.figure()
    for x,y,c,m in zip(range(n_samples), MD, colors, markers):
        plt.scatter(x,y,color=c,marker=m)
    plt.xlim(0, n_samples)
    plt.xlabel("Recording Number")
    plt.ylabel("Mahalanobis Distance")
    gl = plt.scatter([],[], marker='+', color='red')
    gll = plt.scatter([],[], marker='.', color='black')
    plt.legend((gl,gll),("outliers","non-outliers"),loc='upper left')
    if figurename is not None:
        plt.savefig(figurename, bbox_inches='tight')


if __name__ == '__main__':
    # load data
    data, predictions = pickle.load(open(os.path.join('data','feature_space.pickle'),'rb'))
    real_Y, pred_Y = predictions
    ldadata, labels, audiolabels = data
    
    # confusion matrix
    conf_matrix = metrics.confusion_matrix(real_Y, pred_Y, labels=numpy.unique(real_Y))
    plot_conf_matrix(conf_matrix, labels=numpy.unique(real_Y), figurename="Confusion_Matrix_GCP")
    
    # get outliers
    features, classlabels, audiolabels = average_frames(ldadata, labels, audiolabels)
    outlierpercent, y_pred, MD = get_outliers(features, chi2thr=0.995)
    plot_outliers(MD, y_pred, figurename="Outliers_GCP")
    
    # print outliers per country
    counts = Counter(classlabels[y_pred])
    for label in numpy.unique(classlabels):
        if not counts.has_key(label):
            counts.update({label:0})
    print counts
