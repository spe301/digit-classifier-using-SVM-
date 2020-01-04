#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
digits = load_digits()
dir(digits)


# In[82]:


digits.data


# In[83]:


digits.target_names


# In[84]:


df = pd.DataFrame(digits.data,digits.target)
df.head()


# In[85]:


df['target'] = digits.target
df.head(20)


# In[86]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)


# In[87]:


len(X_train)


# In[88]:


len(X_test)


# In[140]:


import numpy as np
from sklearn.svm import SVC
clf = SVC(C=10, gamma='scale', kernel='rbf')
clf.fit(X_train, y_train)


# In[141]:


clf.score(X_test, y_test)


# In[142]:


import matplotlib.pyplot as plt


# In[143]:


digits.data[0]


# In[144]:


digits.images[0]


# In[145]:


plt.gray() 
plt.matshow(digits.images[8]) 
plt.show()


# In[146]:


print(digits.target.shape)
print(digits.target)


# In[153]:


def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 5
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()


# In[154]:


plot_multi(0)


# In[ ]:




