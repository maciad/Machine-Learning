#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sklearn.neighbors
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


# In[2]:


size = 300
X = np.random.rand(size)*5 - 2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8 - 4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv', index=False)
df.plot.scatter(x='x', y='y')


# In[3]:


X = X.reshape(-1,1)
y = y.reshape(-1,1)


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[5]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred_lin_train = lin_reg.predict(X_train)
y_pred_lin_test = lin_reg.predict(X_test)


# In[6]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_lin_train, color='red')
plt.show


# In[7]:


knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)

knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train, y_train)

y_pred_knn_3_train = knn_3_reg.predict(X_train)
y_pred_knn_5_train = knn_5_reg.predict(X_train)
y_pred_knn_3_test = knn_3_reg.predict(X_test)
y_pred_knn_5_test = knn_5_reg.predict(X_test)


# In[8]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_knn_3_train, 'o', color='red')
plt.show()


# In[9]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_knn_5_train, 'o', color='red')
plt.show()


# In[10]:


poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly_2 = poly_features_2.fit_transform(X_train)
lin_reg_poly_2 = LinearRegression()
lin_reg_poly_2.fit(X_poly_2, y_train)

y_pred_poly_2_train = lin_reg_poly_2.predict(poly_features_2.fit_transform(X_train))
y_pred_poly_2_test = lin_reg_poly_2.predict(poly_features_2.fit_transform(X_test))


# In[11]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_poly_2_train,'o', color='red')
plt.show()


# In[12]:


poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly_3 = poly_features_3.fit_transform(X_train)
lin_reg_poly_3 = LinearRegression()
lin_reg_poly_3.fit(X_poly_3, y_train)

y_pred_poly_3_train = lin_reg_poly_3.predict(poly_features_3.fit_transform(X_train))
y_pred_poly_3_test = lin_reg_poly_3.predict(poly_features_3.fit_transform(X_test))


# In[13]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_poly_3_train,'o', color='red')
plt.show()


# In[14]:


poly_features_4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly_4 = poly_features_4.fit_transform(X_train)
lin_reg_poly_4 = LinearRegression()
lin_reg_poly_4.fit(X_poly_4, y_train)

y_pred_poly_4_train = lin_reg_poly_4.predict(poly_features_4.fit_transform(X_train))
y_pred_poly_4_test = lin_reg_poly_4.predict(poly_features_4.fit_transform(X_test))


# In[15]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_poly_4_train,'o', color='red')
plt.show()


# In[16]:


poly_features_5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly_5 = poly_features_5.fit_transform(X_train)
lin_reg_poly_5 = LinearRegression()
lin_reg_poly_5.fit(X_poly_5, y_train)

y_pred_poly_5_train = lin_reg_poly_5.predict(poly_features_5.fit_transform(X_train))
y_pred_poly_5_test = lin_reg_poly_5.predict(poly_features_5.fit_transform(X_test))


# In[17]:


plt.scatter(X_train, y_train)
plt.plot(X_train, y_pred_poly_5_train,'o', color='red')
plt.show()


# In[18]:


mse_lin_train = mean_squared_error(y_train, y_pred_lin_train)
mse_lin_test = mean_squared_error(y_test, y_pred_lin_test)

mse_knn_3_train = mean_squared_error(y_train, y_pred_knn_3_train)
mse_knn_3_test = mean_squared_error(y_test, y_pred_knn_3_test)

mse_knn_5_train = mean_squared_error(y_train, y_pred_knn_5_train)
mse_knn_5_test = mean_squared_error(y_test, y_pred_knn_5_test)

mse_poly_2_train = mean_squared_error(y_train, y_pred_poly_2_train)
mse_poly_2_test = mean_squared_error(y_test, y_pred_poly_2_test)

mse_poly_3_train = mean_squared_error(y_train, y_pred_poly_3_train)
mse_poly_3_test = mean_squared_error(y_test, y_pred_poly_3_test)

mse_poly_4_train = mean_squared_error(y_train, y_pred_poly_4_train)
mse_poly_4_test = mean_squared_error(y_test, y_pred_poly_4_test)

mse_poly_5_train = mean_squared_error(y_train, y_pred_poly_5_train)
mse_poly_5_test = mean_squared_error(y_test, y_pred_poly_5_test)


# In[19]:


df2 = pd.DataFrame([[mse_lin_train, mse_lin_test],
                   [mse_knn_3_train, mse_knn_3_test],
                   [mse_knn_5_train, mse_knn_5_test],
                   [mse_poly_2_train, mse_poly_2_test],
                   [mse_poly_3_train, mse_poly_3_test],
                   [mse_poly_4_train, mse_poly_4_test],
                   [mse_poly_5_train, mse_poly_5_test]],

                  columns=['train_mse','test_mse'],
                  index=['lin_reg', 'knn_3_reg', 'knn_5_reg', 'poly_2_reg', 'poly_3_reg', 'poly_4_reg', 'poly_5_reg'])


# In[20]:


with open('mse.pkl', 'wb') as plik:
    pickle.dump(df2, plik)


# In[21]:


ans = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), 
       (lin_reg_poly_2, poly_features_2), (lin_reg_poly_3, poly_features_3),
       (lin_reg_poly_4, poly_features_4), (lin_reg_poly_5, poly_features_5)]


# In[22]:


with open('reg.pkl', 'wb') as plik2:
    pickle.dump(ans, plik2)

