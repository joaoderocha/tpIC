#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None


# In[2]:


dataset = pd.read_csv(r'C:\Users\montr\Desktop\Curso\2019-2\Trabalho Intel. Comp\train.csv')


# In[5]:


X = dataset.drop(columns="Survived")
y = dataset.iloc[:,2]


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[12]:





# In[ ]:




