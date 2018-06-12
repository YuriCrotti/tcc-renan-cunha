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
import numpy as np

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
    g_search = GridSearchCV(model, parameters, cv=4, scoring='accuracy')
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

X_train = []
X_test = []
y_train = []
y_test = []
sfm = {}

test_size = 0.2
seed = 1

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size)


selected_features_names = []
selected_features_ids = []
times = []
for i in range(10):
    start = time.clock()
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = model_selection.train_test_split(X_train, y_train, test_size=test_size)
    
    clf = LassoCV()
    sfm = SelectFromModel(clf)
    sfm.fit(X_train_fs, y_train_fs)
    n_features = sfm.transform(X_train_fs).shape[1]
    _X_train_fs = sfm.transform(X_train_fs)
    features_columns = dfX.columns
    for i in range(_X_train_fs.shape[1]):
        column_fs = _X_train_fs[:, i]
        for j in range(X_train_fs.shape[1]):
            column_or = X_train_fs[:, j]
            
            # this feature is selected
            if (column_fs == column_or).all():
                selected_features_names.append(features_columns[j])
                selected_features_ids.append(j)
    time_fs = (time.clock() - start)
    times.append(time_fs)

times_mean = np.mean(np.array(times))
times_std = np.std(np.array(times))

top_features_ids = set(selected_features_ids)
top_features_names = set(selected_features_names)
top_features_names_array = []
top_features_ids_array = []

binCounts = np.bincount(selected_features_ids)
ii = np.nonzero(binCounts)[0]
counts = zip(ii,binCounts[ii]) 

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

barlist = plt.bar(range(len(np.unique(selected_features_ids))), binCounts[ii], color = 'green', align='center', width = 0.6)
plt.title('Ocorrência das características selecionadas')
plt.xticks(range(len(counts)), ii)
plt.xlabel('Indíce da característica')
plt.ylabel('Quantidade')

# put
for i in range(len(top_features_ids)):
    if binCounts[ii][i] < 5:
        barlist[i].set_color('r')
    else:
        top_features_names_array.append(featuresNames[counts[i][0]])
        top_features_ids_array.append(counts[i][0])
    

# transform X_train and X_test
X_train_top = X_train[:, top_features_ids_array]
X_test_top = X_test[:, top_features_ids_array]


#X_train_fs = X_train
#X_test_fs = X_test

# test svm classifier
parameters_svm = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[1, 10, 100], 'gamma': [0.001, 0.01, 0.1], 'probability': [True]}
start = time.clock()
best_svm = get_best_model(SVC(), parameters_svm, X_train_top, y_train)
time_train_svm = (time.clock() - start)
start = time.clock()
score_test_svm = best_svm.score(X_test_top, y_test)
time_test_svm = (time.clock() - start)

# test knn classifier
parameters_knn = {'n_neighbors':[2, 10, 5, 20, 50, 100]}
start = time.clock()
best_knn = get_best_model(KNeighborsClassifier(), parameters_knn, X_train_top, y_train)
time_train_knn = (time.clock() - start)
start = time.clock()
score_test_knn = best_knn.score(X_test_top, y_test)
time_test_knn = (time.clock() - start)

# test LDA classifier
parameters_lda = {'solver':('svd', 'lsqr', 'eigen')}
start = time.clock()
best_lda = get_best_model(LinearDiscriminantAnalysis(), parameters_lda, X_train_top, y_train)
time_train_lda = (time.clock() - start)
start = time.clock()
score_test_lda = best_lda.score(X_test_top, y_test)
time_test_lda = (time.clock() - start)

# test decision tree classifier 
parameters_dtree = {'criterion':('gini', 'entropy')}
start = time.clock()
best_dtree = get_best_model(DecisionTreeClassifier(), parameters_dtree, X_train_top, y_train)
time_train_dtree = (time.clock() - start)
start = time.clock()
score_test_dtree = best_dtree.score(X_test_top, y_test)
time_test_dtree = (time.clock() - start)

# test logist regression classifier 
parameters_lreg = {'penalty':('l1', 'l2')}
start = time.clock()
best_lreg = get_best_model(LogisticRegression(), parameters_lreg, X_train_top, y_train)
time_train_lreg = (time.clock() - start)
start = time.clock()
score_test_lreg = best_lreg.score(X_test_top, y_test)
time_test_lreg = (time.clock() - start)

best = best_lda
print("Best parameters set found on development set:")
print(best.best_params_)

print("Grid scores on development set:")
means = best.cv_results_['mean_test_score']
means_fit_time = best.cv_results_['mean_fit_time']
means_score_time = best.cv_results_['mean_score_time']
stds = best.cv_results_['std_test_score']

results = zip(means, stds, means_fit_time, means_score_time,  best.cv_results_['params'])
results.sort(key = lambda tup: tup[0])
for mean, std, fit_time, score_time, params in results:
    print("%0.5f (+/-%0.05f) for %r time: %0.5f"
          % (mean, std * 2, params, fit_time + score_time))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from scipy import interp
import matplotlib.patches as patches
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
    
#plt.plot(*roc_curve(y_test, best_knn.predict_proba(X_test_fs)[:,1])[:2])
#plt.figure()
#plt.plot(*roc_curve(y_test, best_svm.predict_proba(X_test_fs)[:,1])[:2])


cv = StratifiedKFold(n_splits=4,shuffle=True)
fig1 = plt.figure(figsize=[12,12])
ax1 = fig1.add_subplot(111,aspect = 'equal')
#ax1.add_patch(
#    patches.Arrow(0.45,0.5,-0.25,0.25,width=0.3,color='green',alpha = 0.5)
#    )
#ax1.add_patch(
#    patches.Arrow(0.5,0.45,0.25,-0.25,width=0.3,color='red',alpha = 0.5)
#    )

tprs = []
aucs = []
mean_fpr = np.linspace(0,1,100)

clf_plot = best_knn

X_roc = pd.DataFrame(columns = selected_features_names, data = X_train_fs)
y_roc = pd.DataFrame(columns = ['label'], data = y_train)

i = 1
for train,test in cv.split(X_roc, y_roc):
    prob = clf_plot.fit(X_roc.iloc[train],y_roc.iloc[train]).predict_proba(X_roc.iloc[test])[:,1]
    fpr, tpr, t = roc_curve(y_roc.iloc[test], prob)
    tprs.append(interp(mean_fpr, fpr, tpr))
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    print(fpr)
    print(tpr)
    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i, roc_auc))
    i= i+1
    
#plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='blue',
         label=r'ROC Média (AUC = %0.4f )' % (mean_auc),lw=2, alpha=1)

plt.ylabel('Sensibilidade')
plt.xlabel('1 - Especificidade')
plt.title('ROC - K-Nearest Neighbors')
plt.legend(loc="lower right")
#plt.text(0.32,0.7,'Area com maior acurácia',fontsize = 12)
#plt.text(0.63,0.4,'Área com menor acurácia',fontsize = 12)
plt.show()
