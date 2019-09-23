# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 14:32:17 2015

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""

import numpy
import pandas
import os
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler


def load_data(audiolist=None, metadatacsv=None, csvlist=None, headermeta=False, sepcsv=','):
    """ load metadata, path to audio and csv files
    """
    dfcsv = pandas.read_csv(csvlist, header=None, sep=sepcsv)
    dfaudio = pandas.read_csv(audiolist, header=None, sep=sepcsv)
    df = pandas.read_csv(metadatacsv, engine="c", header=headermeta)
    df['Csv'] = dfcsv
    df['Audio'] = dfaudio
    return df


def load_frames(df=None, testclass=None, K=None, nmf_pb=False, delta_mfcc=False, 
                align_chroma=False, average_frames=False, standard_frames=True):
    """ load frame-based features from csv files and post process if needed
    """
    # initialize
    framelist = []
    nfiles = df.shape[0]
    jcsv = df.columns.get_loc("Csv")
    if testclass is not None:
        jclass = df.columns.get_loc(testclass)
    jaudio = df.columns.get_loc("Audio")
    
    # loop over all files and load and features
    for i in range(nfiles):
        if not os.path.exists(df.iat[i, jcsv]):
            continue
        print "loading file " + str(i+1) + " of " + str(nfiles) + " files"
        frame_feat = pandas.read_csv(df.iat[i, jcsv], engine="c", header=None)
        nframes = frame_feat.shape[0]

        # some post processing            
        if K is not None:
            # average scale transform over K bands
            mean_bands = mean_K_bands(frame_feat.get_values(), K)
            frame_feat =  pandas.DataFrame(mean_bands)
        if nmf_pb:
            # decompose pitch bihitogram via NMF
            nmf_decomp = nmf_decomposition(frame_feat.get_values())
            frame_feat = pandas.DataFrame(nmf_decomp)
        if delta_mfcc:
            # append deltas to mfccs
            delta = add_delta_mfcc(frame_feat.get_values())
            frame_feat = pandas.DataFrame(delta)
        if align_chroma:
            # align chroma to bin of max magnitude
            ff = frame_feat.get_values()
            maxind = numpy.argmax(numpy.sum(ff, axis=0))
            ff = numpy.roll(ff, -maxind, axis=1)
            frame_feat = pandas.DataFrame(ff)
        if average_frames:
            # average frames over 8-sec windows
            ave_frames = average_local_frames(frame_feat.get_values().T).T
            frame_feat = pandas.DataFrame(ave_frames)
            nframes = frame_feat.shape[0]  # update number of frames
        if standard_frames:
            # standardize rows (frames)
            ff = frame_feat.T
            frame_feat = StandardScaler().fit(ff).transform(ff).T
        
        # append class and audio label to data frame
        if testclass is not None:
            frame_feat[testclass] = numpy.repeat(df.iat[i, jclass], nframes)
        frame_feat['Audio'] = numpy.repeat(df.iat[i, jaudio], nframes)
        framelist.append(frame_feat)
    dfframes = pandas.concat(framelist)
    return dfframes


def average_local_frames(frames, win2sec=8):
    """ compute frame average and std over window of length win2sec
    """
    sr = 44100/float(round(0.005*44100))  # default sr
    win2 = int(round(win2sec*sr))
    hop2 = int(round(0.5*sr))
    nbins, norigframes = frames.shape
    if norigframes<win2:
        nframes = 1
    else:
        nframes = int(1+numpy.floor((norigframes-win2)/float(hop2)))
    aveframes = numpy.empty((nbins+nbins, nframes))
    # loop over all 8-sec frames
    for i in range(nframes):
        meanf = numpy.mean(frames[:, (i*hop2):min((i*hop2+win2), norigframes)], axis=1)
        stdf = numpy.std(frames[:, (i*hop2):min((i*hop2+win2), norigframes)], axis=1)
        aveframes[:,i] = numpy.concatenate((meanf,stdf))
    return aveframes


def nmf_decomposition(frames):
    """ nmf decomposition for pitch bihistogram descriptor
    """    
    nfr, nbins = frames.shape
    npc = 2
    # assume structure of input is nframes x (nbins*nbins)
    nb = int(numpy.sqrt(nbins))  
    newframes = numpy.empty((nfr, (nb+nb)*npc))
    for fr in range(nfr):                
        pb = numpy.reshape(frames[fr, :], (nb, nb)).T
        nmfmodel = NMF(n_components=npc).fit(pb)
        W = nmfmodel.transform(pb)
        H = nmfmodel.components_.T
        newframes[fr, :, None] = numpy.concatenate((W, H)).flatten()[:, None]
    return newframes


def mean_K_bands(frames, K=40):
    """ average over K mel bands for scale transform descriptor
    """
    [F, P] = frames.shape
    B = 40  # default 40 mel bands
    Pproc = int((P/40)*K)
    procframes = numpy.zeros([F, Pproc])
    niters = int(B/K)
    nbins = P/B  # must be 200 bins
    for k in range(K):
        for j in range(k*niters, (k+1)*niters):
            procframes[:, (k*nbins):((k+1)*nbins)] += frames[:, (j*nbins):((j+1)*nbins)]
    procframes /= float(niters)
    return procframes


def add_delta_mfcc(frames):
    """ append mfccs with deltas
    """    
    f_diff = numpy.diff(frames, axis=0)
    f_delta = numpy.concatenate((f_diff, f_diff[None, -1, :]), axis=0)
    frames = numpy.concatenate([frames, f_delta], axis=1)
    return frames