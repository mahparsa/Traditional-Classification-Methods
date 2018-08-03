
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.svm import NuSVC
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics



##============================Load Data===================================================



df_features = pd.read_csv('EEG.csv')
df_features.head()

X = df_features.values
Y= X[:,14]
X=X[:,0:13]
X_train, X_test, y_train, y_test = train_test_split(X, Y)

##============================Extremely Randomized Trees=====================================

clf_ERT = ExtraTreesClassifier(n_estimators=203, max_depth=None, min_samples_split=7, random_state=0)
###########################################################################################

clf_ERT.fit(X_train, y_train)
predicted_ERT = clf_ERT.predict(X_test)

##============================Scores=======================================================

train_score_ERT = clf_ERT.score(X_train, y_train)
test_score_ERT = clf_ERT.score(X_test, y_test)
print("Train score :", train_score_ERT)
print("Test score :", test_score_ERT)

############################################################################################



predicted_proba_ERT = clf_ERT.predict_proba(X_test)[:,1]
y_pred=predicted_ERT

from sklearn.metrics import accuracy_score #works
import matplotlib.pyplot as plt
fpr, tpr, thresholds = metrics.roc_curve(np.transpose(y_test), np.transpose(predicted_proba_ERT))


print("The Area Under an ROC Curve :", roc_auc_score(y_test,predicted_proba_ERT))

lw=2
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))



plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('M')
plt.legend(loc="lower right")
plt.show()

