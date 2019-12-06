# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 13:18:50 2016

@author: mariapanteli

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation; either version 2 of the License, or (at your option) any later 
version. See the file COPYING included with this distribution for more information.

"""

import os
import numpy
import sys
sys.path.append('util')
import mfccs as mf
import scale_transform as st
import pitch_bihist as pb
import smoothiecore as s


def extract_selected_features(filename=None):
    """ extracts mfcc, scale transform, pitch bihistogram and chroma features
        for audio in filename. 
        Returns each feature as a numpy array with nfeatures x nframes.
    """
    mfc = mf.MFCCs()
    opm = st.ScaleTransform()
    pbh = pb.PitchBihist()
    mfccs = mfc.get_mfccs(filename=filename)
    scalet = opm.get_scale_transform(filename=filename)
    pitchb = pbh.get_pitchbihist(filename=filename)
    chroma, _ = s.get_chroma(filename=filename)
    return mfccs, scalet, pitchb, chroma

  
def extract_features_for_filelist(filelist=None, write_output=False, output_dir='csvfiles'):
    """ extracts selected features for each file in filelist. 
        Set write_output to True to write feature csv files. 
    """    
    for filename in filelist:
        mfccs, scalet, pitchb, chroma = extract_selected_features(filename)
        if write_output:
            csvname = os.path.splitext(os.path.basename(filename))[0] + '.csv'
            # write frames as rows
            numpy.savetxt(os.path.join(output_dir, 'timbre', csvname), mfccs.T, delimiter=',', fmt='%10.12f')
            numpy.savetxt(os.path.join(output_dir, 'rhythm', csvname), scalet.T, delimiter=',', fmt='%10.12f')
            numpy.savetxt(os.path.join(output_dir, 'melody', csvname), pitchb.T, delimiter=',', fmt='%10.12f')
            numpy.savetxt(os.path.join(output_dir, 'harmony', csvname), chroma.T, delimiter=',', fmt='%10.12f')


if __name__ == '__main__':
    with open(os.path.join('data', 'audiolist.txt'), 'rb') as f:
        filelist = [line.strip('\n') for line in f.readlines()]
    extract_features_for_filelist(filelist, write_output=True)
