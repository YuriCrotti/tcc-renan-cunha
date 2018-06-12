#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 09:37:47 2018

@author: bara
"""

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
import time
import numpy as np


def plot_data_histograms(data, bins):
        """
        Plot multiple data histograms in parallel
        :param data : a set of data to be plotted
        :param bins : the number of bins to be used
        :param color : teh color of each data in the set
        :param label : the label of each color in the set
        :param file_path : the path where the output will be save
        """
        plt.figure()
        plt.hist(data, bins, normed=1, alpha=0.75)
        plt.legend(loc='upper right')

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

df = pd.read_csv("dataset.csv")

dfX = df.ix[:, featuresNames]
dfY = df.ix[:, ['label']]

# separe inputs and outputs
X = dfX.values
y = dfY.values

test_size = 0.2
seed = 1
selected_features_names = []
selected_features_ids = []
times = []
for i in range(10):
    start = time.clock()
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)
    
    clf = LassoCV()
    sfm = SelectFromModel(clf)
    sfm.fit(X_train, y_train)
    n_features = sfm.transform(X_train).shape[1]
    X_train_fs = sfm.transform(X_train)
    features_columns = dfX.columns
    for i in range(X_train_fs.shape[1]):
        column_fs = X_train_fs[:, i]
        for j in range(X_train.shape[1]):
            column_or = X_train[:, j]
            
            # this feature is selected
            if (column_fs == column_or).all():
                selected_features_names.append(features_columns[j])
                selected_features_ids.append(j)
    time_fs = (time.clock() - start)
    times.append(time_fs)
    
plt.plot(range(10), times)
            
#data = selected_features_ids
#s = set(selected_features_ids)
#ss = set(selected_features_names)
#
#y = np.bincount(selected_features_ids)
#ii = np.nonzero(y)[0]
#counts = zip(ii,y[ii]) 
#
#
#barlist = plt.bar(range(len(np.unique(selected_features_ids))), y[ii], color = 'green', align='center', width = 0.6)
#plt.title('Frequência das características selecionadas')
#plt.xticks(range(len(counts)), ii)
#plt.xlabel('Indíce da característica')
#plt.ylabel('Frequência')
#
#
#
#for i in range(len(s)):
#    if y[ii][i] < 5:
#        barlist[i].set_color('r')
