traie#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 12:32:09 2018

@author: bara
"""

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# extract features from audio files
[F, labels, files] = audioFeatureExtraction.dirsWavFeatureExtraction(['samples/good', 'samples/bad'], 1, 1, 0.050, 0.025)

featuresNames = [
    'zcr_mean', 'energy_mean', 'entropy_mean', 'spectral_centroid_mean', 'spectral_spread_mean', 'spectral_entropy_mean', 'spectral_flux_mean',
    'spectral_rolloff_mean','mfcc_1_mean','mfcc_2_mean','mfcc_3_mean','mfcc_4_mean','mfcc_5_mean','mfcc_6_mean','mfcc_7_mean','mfcc_8_mean',
    'mfcc_9_mean', 'mfcc_10_mean','mfcc_11_mean','mfcc_12_mean','mfcc_13_mean','chroma_1_mean','chroma_2_mean','chroma_3_mean','chroma_4_mean',
    'chroma_5_mean','chroma_6_mean','chroma_7_mean','chroma_8_mean','chroma_9_mean','chroma_10_mean','chroma_11_mean','chroma_12_mean',
    'chroma_deviation_mean',
    
    'zcr_std', 'energy_std', 'entropy_std', 'spectral_centroid_std', 'spectral_spread_std', 'spectral_entropy_std', 'spectral_flux_std',
    'spectral_rolloff_std','mfcc_1_std','mfcc_2_std','mfcc_3_std','mfcc_4_std','mfcc_5_std','mfcc_6_std','mfcc_7_std','mfcc_8_std',
    'mfcc_9_std', 'mfcc_10_std','mfcc_11_std','mfcc_12_std','mfcc_13_std','chroma_1_std','chroma_2_std','chroma_3_std','chroma_4_std',
    'chroma_5_std','chroma_6_std','chroma_7_std','chroma_8_std','chroma_9_std','chroma_10_std','chroma_11_std','chroma_12_std',
    'chroma_deviation_std'
]

df_good = pd.DataFrame(data = list(F[0]), columns = featuresNames)
df_good['label'] = 1
df_bad = pd.DataFrame(data = list(F[1]), columns = featuresNames)
df_bad['label'] = 0

df = pd.DataFrame(data = [], columns = featuresNames)
df = df.append(df_good)
df = df.append(df_bad)

df.to_csv('dataset.csv', index = False)