
"""
Created on Fri Jul 13 21:00:47 2018

@author: mp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import read_csv
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

##============================Load Data===================================================

df_features = pd.read_csv('dataset/features_train.csv').fillna(0)
df_features.head()
df_target = pd.read_csv("dataset/performance_train.csv")
df_target.head()
X = df_features.values
y = df_target.loc[:,"Default"].values
X_train, X_test, y_train, y_test = train_test_split(X, y)

##============================Random Forest Classifier=====================================

clf_RF = RandomForestClassifier(n_estimators=128, max_depth=None,max_features='log2', min_samples_split=3, random_state=0)

############################################################################################
#"A random forest is an ensemble method of some decision tree classifiers. 
#"Each decision tree classifier is examined on a sub-sample of the dataset. 
#"The classification results obtained by using averaging of the result of each classifier. 
#"Random forests have shown that they can improve the predictive accuracy and overcome over-fitting.  
#"The adjustable parameters of the random forest method are 
#"1) n_estimators: determine the number of trees in the forest. 
#"2) max_features determines the random subsets of features to consider when splitting a node.
### n_estimators is integer and indicate the number of trees. The defulat value is 10. 
##The number of trees in the forest
##"max_features =”auto”(default), "sqrt" (sqrt(n_features)), "log2" (log2(n_features))
##max_depth : integer or None, optional (default=None)
#"determines the maximum depth of the tree.  
#"None means then nodes are expanded until all leaves are pure 
#"or until all leaves contain less than min_samples_split samples.
############################################################################################

clf_RF.fit(X_train, y_train)
predicted_RF = clf_RF.predict(X_test)

##============================Scores=======================================================

train_score_RF = clf_RF.score(X_train, y_train)
test_score_RF = clf_RF.score(X_test, y_test)
print("Train score RandomForest :", train_score_RF)
print("Test score RandomForest:", test_score_RF)

###########################################################################################
##the best results: 
#Train score RandomForest : 0.9996638655462184
#Test score RandomForest: 0.8638655462184874
