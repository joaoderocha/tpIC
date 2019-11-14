#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None


# In[46]:


dataset = pd.read_csv(r'C:\Users\montr\Desktop\Curso\2019-2\Trabalho Intel. Comp\train.csv')


# In[47]:


dataset["Age"].fillna(value=dataset["Age"].mean(),inplace=True)
dataset["Age"] = dataset["Age"].round(0)
gender = {'female': 1,'male': 0}
dataset["Sex"] = [gender[item] for item in dataset["Sex"]]
dataset


# In[57]:


X = dataset.drop(columns={"Name", "Survived", "Ticket", "Cabin", "Embarked"})
y = dataset.iloc[:,1]


# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[61]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)

clf1 = clf.predict(X_test)
clf.score(X_test,y_test)


# In[ ]:




