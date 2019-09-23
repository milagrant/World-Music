# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:26:10 2016

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""
"""Pitch bihistogram descriptor from chroma representation"""

import numpy
import scipy.signal

import smoothiecore as s

class PitchBihist:
    def __init__(self):
        self.y = None
        self.sr = None
        self.chroma = None
        self.chromasr = None
        self.bihist = None

    def bihist_from_chroma(self, filename='test.wav'):
        self.chroma, self.chromasr = s.get_chroma(filename=filename)
        
        # second frame decomposition
        win2 = int(round(8 * self.chromasr))  # default win2 size 8 sec.
        hop2 = int(round(0.5 * self.chromasr))  # default hop2 size 0.5 sec.
        n_bins, n_chroma_frames = self.chroma.shape
        n_frames = max(1, int(1 + numpy.floor((n_chroma_frames-win2) / float(hop2))))
        bihistframes = numpy.empty((n_bins * n_bins, n_frames))
        
        # loop over all 8-sec frames
        for i in range(n_frames):
            frame = self.chroma[:, (i*  hop2):min((i * hop2 + win2), n_chroma_frames)]
            bihist = self.bihistogram(frame)
            bihist = numpy.reshape(bihist, -1)
            bihistframes[:, i] = bihist
        self.bihist = bihistframes

    def bihistogram(self, spec, winsec=0.5, align=True):
        win = int(round(winsec * self.chromasr))
        ker = numpy.concatenate([numpy.zeros((win, 1)), numpy.ones((win+1, 1))], axis=0)
        spec = spec.T  # transpose to have frames as rows in convolution

        # energy threshold
        thr = 0.3 * numpy.max(spec)
        spec[spec < max(thr, 0)] = 0

        # transitions via convolution
        tra = scipy.signal.convolve2d(spec, ker, mode='same')
        tra[spec > 0] = 0

        # multiply with original
        B = numpy.dot(tra.T, spec)

        # normalize
        mxB = numpy.max(B)
        mnB = numpy.min(B)
        if mxB != mnB:  # avoid division by 0
            B = (B - mnB) / float(mxB-mnB)

        # circularly shift to highest magnitude
        if align:
            ref = numpy.argmax(numpy.sum(spec, axis=0))
            B = numpy.roll(B, -ref, axis=0)
            B = numpy.roll(B, -ref, axis=1)
        return B

    def get_pitchbihist(self, filename='test.wav'):
        self.bihist_from_chroma(filename=filename)
        return self.bihist


if __name__ == '__main__':
    pb = PitchBihist()
    pb.get_pitchbihist()
