#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 22 11:13:09 2018

@author: bara
"""

from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import matplotlib.pyplot as plt
import numpy as np

[fs, x_good] = audioBasicIO.readAudioFile("samples/good/5.wav");
x_good = x_good /  (2.**15)
times = np.arange(len(x_good))/float(fs)

plt.subplot(2,1,1); plt.plot(times, x_good); plt.xlabel('Tempo (s)'); plt.ylabel('Amplitude');
plt.show();