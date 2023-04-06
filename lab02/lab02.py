#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


# In[2]:


mnist = fetch_openml('mnist_784', version=1, parser='auto')


# In[3]:


X = mnist['data']
y = mnist['target'].astype(np.uint8)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[5]:


y_train_0 = (y_train == 0)
y_test_0 = (y_test == 0)


# In[6]:


sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_0)


# In[7]:


cva_score_train = cross_val_score(sgd_clf, X_train, y_train_0, n_jobs=-1, scoring='accuracy', cv=3)
score_test =  sgd_clf.score(X_test, y_test_0)
score_train = sgd_clf.score(X_train, y_train_0)


# In[8]:


wyniki1 =[float(score_train), float(score_test)]


# In[9]:


with open('sgd_acc.pkl', 'wb') as plik1:
    pickle.dump(wyniki1, plik1)

with open('sgd_cva.pkl', 'wb') as plik2:
    pickle.dump(cva_score_train, plik2)


# In[10]:


sgd_m_clf = SGDClassifier(random_state=42, n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)
y_train_pred = cross_val_predict(sgd_m_clf, X_train, y_train, cv=3, n_jobs=-1)


# In[11]:


conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)


# In[12]:


with open('sgd_cmx.pkl', 'wb') as plik3:
    pickle.dump(conf_mx, plik3)

