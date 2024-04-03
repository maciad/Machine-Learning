#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler


# ### 1

# In[ ]:


bc = load_breast_cancer()
iris = load_iris()


# In[ ]:


scaler_bc = StandardScaler()
scaler_ir = StandardScaler()


# In[ ]:


X_bc = scaler_bc.fit_transform(bc.data)
X_ir = scaler_ir.fit_transform(iris.data)


# ### 2

# In[ ]:


pca_bc = PCA(n_components=0.9)
pca_ir = PCA(n_components=0.9)

X_bc_pca = pca_bc.fit_transform(X_bc)
X_ir_pca = pca_ir.fit_transform(X_ir)


# ### 3

# In[ ]:


pca_bc_var = np.var(X_bc_pca, axis=0)
pca_bc_var_ratio = list(pca_bc_var / np.sum(pca_bc_var))

pca_ir_var = np.var(X_ir_pca, axis=0)
pca_ir_var_ratio = list(pca_ir_var / np.sum(pca_ir_var))


# In[ ]:


with open('pca_bc.pkl', 'wb') as f:
    pickle.dump(pca_bc_var_ratio, f)

with open('pca_ir.pkl', 'wb') as f:
    pickle.dump(pca_ir_var_ratio, f)


# ### 4

# In[ ]:


idx_bc = [np.argmax(abs(pca_bc.components_[i])) for i in range(pca_bc.n_components_)]
idx_ir = [np.argmax(abs(pca_ir.components_[i])) for i in range(pca_ir.n_components_)]


# In[ ]:


with open('idx_bc.pkl', 'wb') as f:
    pickle.dump(idx_bc, f)
    
with open('idx_ir.pkl', 'wb') as f:
    pickle.dump(idx_ir, f)

