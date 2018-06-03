# -*- coding: utf-8 -*-
"""
Created on Tue May 15 10:37:42 2018

@author: renan
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

def get_best_model(model, parameters, data, target):
    """
    Run GridSearchCV with the specified parameters to find the best configuration for the model
    'model' is the model to tune
    'parameters' is the parameters to test
    'data' is the input
    'target' is the desired output
    
    We use cv (number of folds) = 4 to make sure that the validation set is 20% of the total data 
    Total data = 280
    Train/Validation = 80% of 280 = 224
    
    Train = (224/4)*3 = 168
    Validation = (224/4)*1 = 56
    
    Test = 56
    """
    g_search = GridSearchCV(model, parameters, cv=4)
    g_search.fit(data, target)
    return g_search

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

# read the dataset
df = pd.read_csv("dataset.csv")

dfX = df.ix[:, featuresNames]
dfY = df.ix[:, ['label']]

# separe inputs and outputs
X = dfX.values
y = dfY.values

# split dataset in two sets: train/validation (80%) set and test set (20%)
test_size = 0.3
seed = 1
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)

# run LASSO feature selection
start = time.clock()
clf = LassoCV()
sfm = SelectFromModel(clf)
sfm.fit(X_train, y_train)
n_features = sfm.transform(X_train).shape[1]
time_fs = (time.clock() - start)

# transform train set (remove irrelevant features)
X_train_fs = sfm.transform(X_train)

# transform test set (remove irrelevant features)
X_test_fs = sfm.transform(X_test)

# get selected features names
features_columns = dfX.columns
selected_features = []
for i in range(X_train_fs.shape[1]):
    column_fs = X_train_fs[:, i]
    for j in range(X_train.shape[1]):
        column_or = X_train[:, j]
        
        # this feature is selected
        if (column_fs == column_or).all():
            selected_features.append(features_columns[j]) 


#X_train_fs = X_train
#X_test_fs = X_test

# test svm classifier
parameters_svm = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10, 100, 1000], 'gamma': [0.001, 0.01, 0.1, 1]}
start = time.clock()
best_svm = get_best_model(SVC(), parameters_svm, X_train_fs, y_train)
time_train_svm = (time.clock() - start)
start = time.clock()
score_test_svm = best_svm.score(X_test_fs, y_test)
time_test_svm = (time.clock() - start)

# test knn classifier
parameters_knn = {'n_neighbors':[1, 10, 5, 20, 50, 100]}
start = time.clock()
best_knn = get_best_model(KNeighborsClassifier(), parameters_knn, X_train_fs, y_train)
time_train_knn = (time.clock() - start)
start = time.clock()
score_test_knn = best_knn.score(X_test_fs, y_test)
time_test_knn = (time.clock() - start)

# test LDA classifier
parameters_lda = {'solver':('svd', 'lsqr', 'eigen')}
start = time.clock()
best_lda = get_best_model(LinearDiscriminantAnalysis(), parameters_lda, X_train_fs, y_train)
time_train_lda = (time.clock() - start)
start = time.clock()
score_test_lda = best_lda.score(X_test_fs, y_test)
time_test_lda = (time.clock() - start)

# test decision tree classifier 
parameters_dtree = {'criterion':('gini', 'entropy')}
start = time.clock()
best_dtree = get_best_model(DecisionTreeClassifier(), parameters_dtree, X_train_fs, y_train)
time_train_dtree = (time.clock() - start)
start = time.clock()
score_test_dtree = best_dtree.score(X_test_fs, y_test)
time_test_dtree = (time.clock() - start)

# test logist regression classifier 
parameters_lreg = {'penalty':('l1', 'l2')}
start = time.clock()
best_lreg = get_best_model(LogisticRegression(), parameters_lreg, X_train_fs, y_train)
time_train_lreg = (time.clock() - start)
start = time.clock()
score_test_lreg = best_lreg.score(X_test_fs, y_test)
time_test_lreg = (time.clock() - start)