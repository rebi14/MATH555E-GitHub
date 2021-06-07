#!/usr/bin/env python
# coding: utf-8

# # Homework 5
# 
# ## Image Processing and Pixel Classification
# 
# 
# This week's homework is about classifying pixels in a sattelite image:

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat
import pandas as pd


# The image below a sattelite image of Salinas Valley, California of 512 pixels by 217 pixels where individual values tells us what is planted in that specific area: 

# In[127]:


salinas_gt = loadmat('Salinas_gt.mat')['salinas_gt']
salinas = loadmat('Salinas.mat')['salinas']
salinas_classes = pd.read_csv('Salinas_classes.csv',sep='\t')['Class ']
plt.imshow(salinas_gt) 


# In[128]:


salinas_gt


# In[129]:


salinas


# In[130]:


salinas_classes


# The main data `salinas` is a tensor (i.e. a multi-dimensional array) of shape (512,217,224). This means we have 224 different greyscale images taken in different wavelengths. Below, I am going to reshape the data into a matrix of shape $(512\cdot 217,224)$ so that each row is a flattened matrix.

# In[123]:


scaler = MinMaxScaler()
salinas_reshaped = scaler.fit_transform(salinas.reshape((512*217,224)))
salinas_gt_reshaped = salinas_gt.reshape(512*217)

plt.imshow(salinas_reshaped[:,0].reshape((512,217)))


# In[125]:


salinas_reshaped


# In[126]:


salinas_gt_reshaped


# In[131]:


from sklearn.model_selection import train_test_split


# In[156]:


X_train, X_test, y_train, y_test = train_test_split(salinas_reshaped,salinas_gt_reshaped)
print(f"X_train: {X_train.shape}\ny_train: {y_train.shape}\nX_test: {X_test.shape}\ny_test: {y_test.shape}") 


# In[138]:


#1.Linear Discriminant Analysis model
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X_train,y_train)


# In[139]:


from sklearn.metrics import accuracy_score

pred_y = clf.predict(X_test)

accuracy = accuracy_score(y_test, pred_y, normalize=True, sample_weight=None)
accuracy


# In[ ]:


#2.Support Vector Machines model


# In[142]:


from sklearn.svm import SVC


# In[143]:


svc=SVC() #Default hyperparameters
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


#3.Logistic Regression model


# In[144]:


from sklearn.linear_model import LogisticRegression


# In[150]:


rfe = LogisticRegression()
rfe.fit(X_train,y_train)
y_pred=rfe.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


#4.Decision Tree model


# In[147]:


from sklearn import tree


# In[151]:


dt2 = tree.DecisionTreeClassifier()
dt2.fit(X_train,y_train)
y_pred=dt2.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))


# In[ ]:


#5.Boosted Tree model


# In[167]:


from xgboost import XGBRegressor
import xgboost as xgb
xgbr = xgb.XGBRegressor(verbosity=0) 


# In[168]:


XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=1, gamma=0,
       importance_type='gain', learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)
xgbr.fit(X_train, y_train)
score = xgbr.score(X_train, y_train)
print("Training score: ", score)



# In[ ]:




