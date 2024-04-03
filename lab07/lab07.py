#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import pickle
from statistics import mean
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix


# In[ ]:


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
mnist.target = mnist.target.astype(np.uint8)
X = mnist["data"]
y = mnist["target"]


# ### 1

# In[ ]:


km_8 = KMeans(n_clusters=8, n_init=5)
y_pred_8 = km_8.fit_predict(X)


# In[ ]:


km_9 = KMeans(n_clusters=9, n_init=5)
y_pred_9 = km_9.fit_predict(X)


# In[ ]:


km_10 = KMeans(n_clusters=10, n_init=5)
y_pred_10 = km_10.fit_predict(X)


# In[ ]:


km_11 = KMeans(n_clusters=11, n_init=5)
y_pred_11 = km_11.fit_predict(X)


# In[ ]:


km_12 = KMeans(n_clusters=12, n_init=5)
y_pred_12 = km_12.fit_predict(X)


# ### 2

# In[ ]:


silhouettes = []
for km in (km_8, km_9, km_10, km_11, km_12):
    silhouettes.append(silhouette_score(X, km.labels_))


# In[ ]:


with open('kmeans_sil.pkl', 'wb') as f:
    pickle.dump(silhouettes, f)


# ### 4

# In[ ]:


conf_matrix = confusion_matrix(y, y_pred_10)


# ### 5

# In[ ]:


s = set()
for row in conf_matrix:
    s.add(np.argmax(row))
l = list(s)


# In[ ]:


with open('kmeans_argmax.pkl', 'wb') as f:
    pickle.dump(l, f)


# ### 6

# In[ ]:


distances = []
for i in X[:300]:
    for j in X:
        if any(i != j):
            dist = np.linalg.norm(i-j)
            distances.append(dist)


# In[ ]:


distances.sort()


# In[ ]:


min_10_dist = distances[:10]


# In[ ]:


with open('dist.pkl', 'wb') as f:
    pickle.dump(min_10_dist, f)


# ### 7

# In[ ]:


s = mean(distances[:3])


# In[ ]:


dbscan = DBSCAN(eps=s)
dbscan_104 = DBSCAN(eps=1.04*s)
dbscan_108 = DBSCAN(eps=1.08*s)


# ### 8

# In[ ]:


dbscan_lengths = []
for db in (dbscan, dbscan_104, dbscan_108):
    db.fit(X)
    n_clusters = len(set(db.labels_))
    dbscan_lengths.append(n_clusters)


# In[ ]:


with open('dbscan_len.pkl', 'wb') as f:
    pickle.dump(dbscan_lengths, f)

