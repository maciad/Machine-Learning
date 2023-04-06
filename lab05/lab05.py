#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import graphviz
from sklearn.tree import export_graphviz
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ## Przygotowanie danych

# In[ ]:


bc = datasets.load_breast_cancer(as_frame=True)

X_train, X_test, y_train, y_test = train_test_split(
    bc.data[['mean texture', 'mean symmetry']], bc.target, test_size=0.2)


# In[ ]:


size = 300
X2 = np.random.rand(size) * 5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y2 = w4*(X2**4) + w3*(X2**3) + w2*(X2**2) + w1*X2 + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X2, 'y': y2})
# df.plot.scatter(x='x', y='y')

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2)


# ## Zadanie 3

# In[ ]:


tree_clf = DecisionTreeClassifier(max_depth=3)
tree_clf.fit(X_train, y_train)


# In[ ]:


graph_file = 'bc'
graph_dot = export_graphviz(
    tree_clf,
    out_file=graph_file,
    feature_names = ['mean texture', 'mean symmetry'],
    class_names=bc.target_names,
    rounded=True,
    filled=True)

# graph = graphviz.Source(graph_dot)
graphviz.render('dot', 'png', graph_file)
os.remove('bc')


# In[ ]:


y_pred_train = tree_clf.predict(X_train)
y_pred_test = tree_clf.predict(X_test)

f1_score_train = f1_score(y_train, y_pred_train)
f1_score_test = f1_score(y_test, y_pred_test)
acc_train = tree_clf.score(X_train, y_train)
acc_test = tree_clf.score(X_test, y_test)


# In[ ]:


output = [tree_clf.max_depth, f1_score_train, f1_score_test,
          acc_train, acc_test]
with open('f1acc_tree.pkl', 'wb') as f:
    pickle.dump(output, f)


# ## zadanie 4

# In[ ]:


tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X2_train.reshape(-1, 1), y2_train)


# In[ ]:


y2_pred = tree_reg.predict(X2.reshape(-1, 1))
plt.scatter(X2_train, y2_train)
plt.plot(X2, y2_pred, 'o', color='red')
plt.show()


# In[ ]:


graph2_file = 'reg'
graph2_dot = export_graphviz(
    tree_reg,
    out_file=graph2_file,
    feature_names = ['x'],
    rounded=True,
    filled=True)

# graph = graphviz.Source(graph_dot)
graphviz.render('dot', 'png', graph2_file)
os.remove('reg')


# In[ ]:


y2_pred_train = tree_reg.predict(X2_train.reshape(-1, 1))
y2_pred_test = tree_reg.predict(X2_test.reshape(-1, 1))

mse_train = mean_squared_error(y2_train, y2_pred_train)
mse_test = mean_squared_error(y2_test, y2_pred_test)


# In[ ]:


output2 = [tree_reg.max_depth, mse_train, mse_test]

with open('mse_tree.pkl', 'wb') as f:
    pickle.dump(output2, f)

