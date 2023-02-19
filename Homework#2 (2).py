#!/usr/bin/env python
# coding: utf-8

# In[95]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import seaborn as sns
import csv
from math import dist
from scipy import stats
import random


iris_data = pd.read_csv("http://theparticle.com/cs/bc/dsci/iris.csv")
print(iris_data.head())


# In[151]:


#iris = load_iris()
#X = iris.data
#y = iris.target
#sns.set(style="ticks")
#df= sns.load_dataset("iris")
sns.pairplot(df, hue="species")
plt.show()


# In[133]:


from sklearn.datasets import load_iris
iris =load_iris()
iris.keys()


# In[150]:


print(iris['DESCR'])


# In[122]:


iris_data.isnull().sum()


# In[112]:


iris_data.duplicated().sum()


# In[113]:


iris=iris_data.drop_duplicates


# In[114]:


X = iris_data.iloc[:, :-1].values
y = iris_data.iloc[:, 4].values


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# In[85]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[86]:


from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)


# In[87]:


y_pred = classifier.predict(X_test)
y_pred


# In[88]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))


# In[99]:


from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(X_train,y_train)
predict2=model_LR.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predict2)*100)


# In[ ]:




