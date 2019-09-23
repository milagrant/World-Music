# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 13:37:00 2015

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""
"""Scale transform descriptor from mel spectrogram"""

import numpy
import librosa
import scipy.signal

class ScaleTransform:
    def __init__(self):
        self.y = None
        self.sr = None
        self.nmels = 40
        self.melspec = None
        self.melsr = None
        self.op = None
        self.opmellin = None

    
    def load_audiofile(self, filename='test.wav', sr=None):
        """Load audio"""
        self.y, self.sr = librosa.load(filename, sr=sr)


    def mel_spectrogram(self, y=None, sr=None):
        """Get mel spectrogram"""
        if self.y is None:
            self.y = y
        if self.sr is None:
            self.sr = sr
        win1 = int(round(0.04 * self.sr))
        hop1 = int(round(win1 / 8.))
        nfft1 = int(2 ** numpy.ceil(numpy.log2(win1)))
        D = numpy.abs(librosa.stft(self.y, n_fft=nfft1, hop_length=hop1, win_length=win1, window=scipy.signal.hamming)) ** 2
        melspec = librosa.feature.melspectrogram(S=D, sr=self.sr, n_mels=self.nmels, fmax=8000)
        melsr = self.sr/float(hop1)
        self.melspec = melspec
        self.melsr = melsr

 
    def post_process_spec(self, melspec=None, medianfilt=True, sqrt=True, diff=True, subtractmean=True, halfwave=True, maxnormal=True):
        """Some post processing of the mel spectrogram"""        
        if self.melspec is None:
            self.melspec = melspec
        if medianfilt:
            ks = int(0.1 * self.melsr) # 100ms kernel size
            if ks % 2 == 0: ks += 1  # ks must be odd
            for i in range(self.nmels):
                self.melspec[i, :] = scipy.signal.medfilt(self.melspec[i, :], kernel_size=ks)
        if sqrt:
            self.melspec = self.melspec ** .5
        if diff:
            # append one frame before diff to keep number of frames the same
            self.melspec = numpy.concatenate((self.melspec, self.melspec[:, -1, None]), axis=1)
            self.melspec = numpy.diff(self.melspec, n=1, axis=1)
        if subtractmean:
            mean = self.melspec.mean(axis=1)
            mean.shape = (mean.shape[0], 1)
            self.melspec = self.melspec - mean
        if halfwave:
            self.melspec[numpy.where(self.melspec < 0)] = 0
        if maxnormal:
            self.melspec = self.melspec / self.melspec.max()


    def onset_patterns(self, melspec=None, melsr=None, center=False):
        """Get rhythm periodicities by applying stft in each mel band"""
        if self.melspec is None:
            self.melspec = melspec
        if self.melsr is None:
            self.melsr = melsr
        win2 = int(round(8 * self.melsr))
        hop2 = int(round(0.5 * self.melsr))
        nfft2 = int(2**numpy.ceil(numpy.log2(win2)))
        
        # some preprocessing for the second frame decomposition
        melspectemp = self.melspec
        if ((nfft2 > win2) and (center is False)):
                # pad the signal by nfft2-win2 so that frame decomposition
                # returns the same number of frames as the pitch_bihist descriptor
                melspectemp = numpy.concatenate([numpy.zeros((self.nmels,int((nfft2 - win2) // 2))),self.melspec, numpy.zeros((self.nmels,int((nfft2 - win2) // 2)))],axis=1)
        if melspectemp.shape[1] < nfft2:
            # if buffer too short pad with zeros to have at least one 8-sec window
            nzeros = nfft2 - melspectemp.shape[1]
            melspectemp = numpy.concatenate([numpy.zeros((self.nmels, int(numpy.ceil(nzeros / 2.)))), melspectemp, numpy.zeros((self.nmels,int(numpy.ceil(nzeros / 2.))))], axis=1)
        temp = numpy.abs(librosa.stft(y=melspectemp[0, :], win_length=win2, hop_length=hop2, n_fft=nfft2, window=scipy.signal.hamming, center=center))
        nframes = temp.shape[1]
        
        # filter periodicities in the range 30-960 bpm
        freqresinbpm = float(self.melsr) / float(nfft2/2.)*60.
        minmag = int(numpy.floor(30. / freqresinbpm))  # min tempo 30bpm
        maxmag = int(numpy.ceil(960. / freqresinbpm))  # max tempo 960 bpm
        magsinds = range(minmag, maxmag)  # indices of selected stft magnitudes

        # loop over all mel_bands and get rhythm periodicities (stft magnitudes)
        nmags = len(magsinds)
        fft2 = numpy.zeros((self.nmels, nmags, nframes))
        for i in range(self.nmels):
            fftmags = numpy.abs(librosa.stft(y=melspectemp[i, :], win_length=win2, hop_length=hop2, n_fft=nfft2, window=scipy.signal.hamming, center=center))
            fftmags = fftmags[magsinds, :]
            fft2[i, :, :] = fftmags
        op = fft2
        self.op = op


    def post_process_op(self, median_filt=True):
        """Some smoothing of the onset patterns"""
        if median_filt:
            hop2 = int(round(0.5 * self.melsr))
            ssr = self.melsr/float(hop2)
            ks = int(0.5 * ssr)  # 100ms kernel size
            if ks % 2 == 0: ks += 1  # ks must be odd
            nmels, nmags, nframes = self.op.shape
            for i in range(nmels):
                for j in range(nframes):
                    self.op[i, :, j] = numpy.convolve(self.op[i, :, j], numpy.ones(ks) / ks, mode='same')


    def mellin_transform(self, op=None):
        """
        Apply mellin transform to remove tempo (scale) information.
        Code adapted from a MATLAB implementation by Andre Holzapfel.
        """
        if self.op is None:
            self.op = op
        nmels, nmags, nframes = self.op.shape
        nmagsout = 200
        u_max = numpy.log(nmags)
        delta_c = numpy.pi / u_max
        c_max = nmagsout
        c = numpy.arange(delta_c, c_max, delta_c)
        k = range(1, nmags)
        exponent = 0.5 - c * 1j

        normMat = 1. / (exponent * numpy.sqrt(2 * numpy.pi))
        normMat.shape = (normMat.shape[0], 1)
        normMat = numpy.repeat(normMat.T, nmels, axis=0)
        kernelMat = numpy.asarray([numpy.power(ki, exponent) for ki in k])
        opmellin = numpy.zeros((nmels, nmagsout, nframes))
        for i in range(nframes):
            self.op[:, -1, i] = 0
            deltaMat = - numpy.diff(self.op[:, :, i])
            mellin = numpy.abs(numpy.dot(deltaMat, kernelMat) * normMat)
            opmellin[:, :, i] = mellin[:, :nmagsout]
        self.opmellin = opmellin


    def post_process_mellin(self, opmellin=None, normFrame=True, aveBands=False):
        """Some post processing of the scale transform"""
        if self.opmellin is None:
            self.opmellin = opmellin
        if aveBands:
            self.opmellin = numpy.mean(self.opmellin, axis=0, keepdims=True)
        nmels, nmags, nframes = self.opmellin.shape
        self.opmellin = self.opmellin.reshape((nmels*nmags, nframes))
        if normFrame:
            min_opmellin = numpy.amin(self.opmellin, axis=0, keepdims=True)
            max_opmellin = numpy.amax(self.opmellin, axis=0, keepdims=True)
            denom = max_opmellin - min_opmellin
            denom[denom==0] = 1 # avoid division by 0 if frame is all 0s-silent
            self.opmellin = (self.opmellin - min_opmellin) / denom


    def get_scale_transform(self, filename='test.wav'):
        """Return scale transform for filename"""
        self.load_audiofile(filename=filename)
        self.mel_spectrogram()
        self.post_process_spec()
        self.onset_patterns()
        self.post_process_op()
        self.mellin_transform()
        self.post_process_mellin()
        return self.opmellin
        

if __name__ == '__main__':
    op = ScaleTransform()
    op.get_scale_transform()
