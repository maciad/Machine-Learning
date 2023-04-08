#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[2]:


bc = datasets.load_breast_cancer(as_frame=True)


# ### 1

# In[3]:


X_train, X_test, y_train, y_test = train_test_split(
    bc.data, bc.target, test_size=0.2)


# ### 2

# In[4]:


log_clf = LogisticRegression()
tree_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()

voting_clf_hard = VotingClassifier(
    estimators=[('tr', tree_clf),('lr', log_clf),('kn', knn_clf)],
    voting='hard',)

voting_clf_soft = VotingClassifier(
    estimators=[('tr', tree_clf),('lr', log_clf),('kn', knn_clf)],
    voting='soft',)


# ### 3 and 4

# In[5]:


scores = []
for clf in (tree_clf, log_clf, knn_clf, voting_clf_hard, voting_clf_soft):
    clf.fit(X_train[['mean texture', 'mean symmetry']], y_train)
    y_pred_train = clf.predict(X_train[['mean texture', 'mean symmetry']])
    y_pred_test = clf.predict(X_test[['mean texture', 'mean symmetry']])
    s1 = accuracy_score(y_train, y_pred_train)
    s2 = accuracy_score(y_test, y_pred_test)
    scores.append((s1,s2))


# In[6]:


classifiers = [log_clf, tree_clf, knn_clf, 
               voting_clf_hard, voting_clf_soft]


# In[7]:


with open('acc_vote.pkl', 'wb') as f1, open('vote.pkl', 'wb') as f2:
    pickle.dump(scores, f1)
    pickle.dump(classifiers, f2)


# ### 5

# In[8]:


bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    bootstrap=True)

bag_clf_50 = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    bootstrap=True, max_samples=0.5)

past_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    bootstrap=False)

past_clf_50 = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=30,
    bootstrap=False, max_samples=0.5)

rand_clf = RandomForestClassifier(n_estimators=30)

adab_clf = AdaBoostClassifier(n_estimators=30)

gb_clf = GradientBoostingClassifier(n_estimators=30)


# ### 6

# In[9]:


scores2 = []
for clf in (bag_clf, bag_clf_50, past_clf, past_clf_50, rand_clf, adab_clf, gb_clf):
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    s1 = accuracy_score(y_train, y_pred_train)
    s2 = accuracy_score(y_test, y_pred_test)
    scores2.append((s1,s2))


# In[10]:


classifiers2 = [bag_clf, bag_clf_50, past_clf, past_clf_50, rand_clf, adab_clf, gb_clf]


# In[11]:


with open('acc_bag.pkl', 'wb') as f1, open('bag.pkl', 'wb') as f2:
    pickle.dump(scores2, f1)
    pickle.dump(classifiers2, f2)


# ### 7 and 8

# In[12]:


bag_clf_sampling = BaggingClassifier(
    DecisionTreeClassifier(),
    n_estimators=30,
    max_features=2,
    bootstrap_features=False,
    max_samples=0.5,
    bootstrap=True)


# In[13]:


bag_clf_sampling.fit(X_train, y_train)
y_pred_train = bag_clf_sampling.predict(X_train)
y_pred_test = bag_clf_sampling.predict(X_test)
score_train = accuracy_score(y_train, y_pred_train)
score_test = accuracy_score(y_test, y_pred_test)


# In[14]:


scores3 = [score_train, score_test]
classifiers3 = [bag_clf_sampling]


# In[15]:


with open('acc_fea.pkl', 'wb') as f1, open('fea.pkl', 'wb') as f2:
    pickle.dump(scores3, f1)
    pickle.dump(classifiers3, f2)


# ### 9

# In[16]:


df = pd.DataFrame()
for i in range(len(bag_clf_sampling.estimators_)):
    selected_features = X_train.columns[bag_clf_sampling.estimators_features_[i]]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    y_pred_train = bag_clf_sampling.estimators_[i].predict(X_train_selected)
    y_pred_test = bag_clf_sampling.estimators_[i].predict(X_test_selected)
    score_train = accuracy_score(y_train, y_pred_train)
    score_test = accuracy_score(y_test, y_pred_test)
    df = df.append(pd.DataFrame([[score_train, score_test, selected_features.tolist()]], columns=['train', 'test', 'features'], index=[i]))


# In[17]:


df.sort_values(by=['test', 'train'], ascending=False, inplace=True)


# In[18]:


with open('acc_fea_rank.pkl', 'wb') as f:
    pickle.dump(df, f)

