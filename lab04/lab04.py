#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# In[ ]:


breast_cancer = datasets.load_breast_cancer(as_frame=True)
iris = datasets.load_iris(as_frame=True)


# In[ ]:


X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    iris.data.iloc[:,[2,3]], iris.target, test_size=0.2)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    breast_cancer.data.iloc[:, [3,4]],
    breast_cancer.target, test_size=0.2)


# In[ ]:


svm_clf_bc = LinearSVC(C=1, loss='hinge')
svm_clf_scaled_bc = Pipeline([('scaler', StandardScaler()),
                              ('linear_svc', LinearSVC(C=1, 
                                                    loss='hinge'))])
svm_clf_bc.fit(X_train_bc, y_train_bc)
svm_clf_scaled_bc.fit(X_train_bc, y_train_bc)


# In[ ]:


score_train_bc = svm_clf_bc.score(X_train_bc, y_train_bc)
score_test_bc = svm_clf_bc.score(X_test_bc, y_test_bc)
score_train_bc_scaled = svm_clf_scaled_bc.score(X_train_bc, y_train_bc)
score_test_bc_scaled = svm_clf_scaled_bc.score(X_test_bc, y_test_bc)


# In[ ]:


accuracy_bc = [score_train_bc, score_test_bc,
               score_train_bc_scaled, score_test_bc_scaled]


# In[ ]:


with open('bc_acc.pkl', 'wb') as file:
    pickle.dump(accuracy_bc, file)


# In[ ]:


svm_clf_i = LinearSVC(C=1, loss='hinge')
svm_clf_scaled_i = Pipeline([('scaler', StandardScaler()),
                      ('linear_svc', LinearSVC(C=1,
                                               loss='hinge'))])
svm_clf_i.fit(X_train_i, y_train_i)
svm_clf_scaled_i.fit(X_train_i, y_train_i)


# In[ ]:


score_train_i = svm_clf_i.score(X_train_i, y_train_i)
score_test_i = svm_clf_i.score(X_test_i, y_test_i)
score_train_i_scaled = svm_clf_scaled_i.score(X_train_i, y_train_i)
score_test_i_scaled = svm_clf_scaled_i.score(X_test_i, y_test_i)


# In[ ]:


accuracy_i = [score_train_i, score_test_i,
              score_train_i_scaled, score_test_i_scaled]


# In[ ]:


with open('iris_acc.pkl', 'wb') as file2:
    pickle.dump(accuracy_i, file2)

