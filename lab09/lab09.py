#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


# In[ ]:


iris = load_iris(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data[['petal length (cm)','petal width (cm)']],
    iris.target,
    test_size=0.8
)


# ### 2.1

# In[ ]:


y_train_0 = (y_train == 0).astype(int)
y_test_0 = (y_test == 0).astype(int)
y_train_1 = (y_train == 1).astype(int)
y_test_1 = (y_test == 1).astype(int)
y_train_2 = (y_train == 2).astype(int)
y_test_2 = (y_test == 2).astype(int)


# In[ ]:


per_clf_0 = Perceptron()
per_clf_1 = Perceptron()
per_clf_2 = Perceptron()


# In[ ]:


per_clf_0.fit(X_train, y_train_0)
per_clf_1.fit(X_train, y_train_1)
per_clf_2.fit(X_train, y_train_2)


# In[ ]:


score_train_0 = per_clf_0.score(X_train, y_train_0)
score_train_1 = per_clf_1.score(X_train, y_train_1)
score_train_2 = per_clf_2.score(X_train, y_train_2)
score_test_0 = per_clf_0.score(X_test, y_test_0)
score_test_1 = per_clf_1.score(X_test, y_test_1)
score_test_2 = per_clf_2.score(X_test, y_test_2)


# In[ ]:


scores = [(score_train_0, score_test_0),
          (score_train_1, score_test_1),
          (score_train_1, score_test_2)]

weights = [(per_clf_0.intercept_[0], per_clf_0.coef_[0][0], per_clf_0.coef_[0][1]),
           (per_clf_1.intercept_[0], per_clf_1.coef_[0][0], per_clf_1.coef_[0][1]),
           (per_clf_2.intercept_[0], per_clf_2.coef_[0][0], per_clf_2.coef_[0][1])]


# In[ ]:


with open('per_acc.pkl', 'wb') as f:
    pickle.dump(scores, f)

with open('per_wght.pkl', 'wb') as f:
    pickle.dump(weights, f)


# ### 2.2

# In[ ]:


X = np.array(
    [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]])
y = np.array(
    [0,
     1,
     1,
     0])


# In[ ]:


per_clf_xor = Perceptron()
per_clf_xor.fit(X, y)


# ### 2.3

# In[ ]:


mlp_clf = MLPClassifier(hidden_layer_sizes=(2,), activation='relu', solver='adam', max_iter=10000)
mlp_clf.fit(X, y)


# In[ ]:


with open('mlp_xor.pkl', 'wb') as f:
    pickle.dump(mlp_clf, f)


# In[ ]:


weights1 = np.array([[1., 1.], [1., 1.]])
weights2 = np.array([[-1.], [1.]])
biases1 = np.array([-1.5, -0.5])
bias2 = np.array([-0.5])


# In[ ]:


mlp_clf.coefs_ = [weights1, weights2]
mlp_clf.intercepts_ = [biases1, bias2]


# In[ ]:


with open('mlp_xor_fixed.pkl', 'wb') as f:
    pickle.dump(mlp_clf, f)

