#!/usr/bin/env python
# coding: utf-8


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
import os
from urllib import request
import tarfile
import gzip


# In[2]:

url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
response = request.urlretrieve(url, "housing.tgz")
with tarfile.open('housing.tgz', 'r') as file:
    file.extractall(path='./data')

with open('./data/housing.csv', 'rb') as f_in, gzip.open('./data/housing.csv.gz', 'wb') as f_out:
    f_out.writelines(f_in)

os.remove('housing.tgz')
os.remove('./data/housing.csv')

df = pd.read_csv('./data/housing.csv.gz')

# In[3]:


df.head()

# In[4]:


df.info()

# In[5]:


df['ocean_proximity'].value_counts()

# In[6]:


df['ocean_proximity'].describe()

# In[7]:


df.hist(bins=50, figsize=(20, 15))
plt.savefig('obraz1.png')

# In[8]:


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(7, 4))
plt.savefig('obraz2.png')

# In[9]:


df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7, 3), colorbar=True,
        s=df['population'] / 100, label='population',
        c='median_house_value', cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')

# In[10]:


df.corr()["median_house_value"].sort_values(ascending=False).reset_index().rename(
    columns={'index': 'atrybut', 'median_house_value': 'wspolczynnik_korelacji'}).to_csv('korelacja.csv', index=False)

# In[11]:


sns.pairplot(df)

# In[12]:


train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
len(train_set), len(test_set)

# In[13]:


train_set.head(10)

# In[14]:


test_set.tail(10)

# In[15]:


train_set.corr()

# In[16]:


test_set.corr()

# In[17]:


with open('train_set.pkl', 'wb') as asdf:
    pickle.dump(train_set, asdf)

# In[18]:


with open('test_set.pkl', 'wb') as asd:
    pickle.dump(test_set, asd)

# In[ ]:
