# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 12:58:07 2015

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""
import librosa
import scipy.signal
import numpy

class MFCCs:
    def __init__(self):
        self.y = None
        self.sr = None
        self.nmels = 40
        self.melspec = None
        self.melsr = None
        self.mfccs = None

    def load_audiofile(self, filename='test.wav', sr=None):
        self.y, self.sr = librosa.load(filename, sr=sr)
    
    def mel_spectrogram(self, y=None, sr=None):
        """ compute mel-scale spectrogram
        """
        if self.y is None:
            self.y = y
        if self.sr is None:
            self.sr = sr
        win1 = int(round(0.04*self.sr))
        hop1 = int(round(win1/8.))
        nfft1 = int(2**numpy.ceil(numpy.log2(win1)))
        D = numpy.abs(librosa.stft(self.y, n_fft=nfft1, hop_length=hop1, win_length=win1, window=scipy.signal.hamming))**2
        melspec = librosa.feature.melspectrogram(S=D, sr=self.sr, n_mels=self.nmels, fmax=8000)  
        melsr = self.sr/float(hop1)
        self.melspec = melspec
        self.melsr = melsr
        
    def calc_mfccs(self, y=None, sr=None):
        """ calculate mfccs
        """
        if self.y is None:
            self.y = y
        if self.sr is None:
            self.sr = sr
        # require log-amplitude
        self.mfccs = librosa.feature.mfcc(S=librosa.amplitude_to_db(self.melspec), n_mfcc=21)
        # remove first component
        self.mfccs = self.mfccs[1:,:]
        
    def get_mfccs(self, filename=None):
        """ calculate and return mfccs
        """
        self.load_audiofile(filename=filename)
        self.mel_spectrogram()
        self.calc_mfccs()
        return self.mfccs

if __name__ == '__main__':
    mfc = MFCCs()
    mfc.get_mfccs()