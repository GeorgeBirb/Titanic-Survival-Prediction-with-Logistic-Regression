#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


train = pd.read_csv('titanic_train.csv')


# In[9]:


train.head()


# In[11]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[14]:


sns.set_style('whitegrid')


# In[16]:


sns.countplot(x='Survived',hue='Sex',data=train, palette='RdBu_r')


# In[17]:


sns.countplot(x='Survived',hue='Pclass',data=train)


# In[18]:


sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[19]:


train['Age'].plot.hist(bins=35)


# In[20]:


train.info()


# In[21]:


sns.countplot(x='SibSp',data=train)


# In[24]:


train['Fare'].hist(bins=40,figsize=(10,4))


# In[31]:


import cufflinks as cf    


# In[32]:


cf.go_offline()


# In[34]:


train['Fare'].iplot(kind='hist',bins=30)


# In[35]:


sns.boxplot(x='Pclass',y='Age',data=train)


# In[43]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1 :
            return 37
        elif Pclass == 2 : 
            return 29
        else :
            return 24
    else:
        return Age


# In[44]:


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


# In[47]:


train.drop('Cabin',axis=1,inplace=True)


# In[48]:


train.head()


# In[50]:


train.dropna(inplace=True)


# In[53]:


sns.heatmap(train.isnull(), yticklabels=False,cbar=False,cmap='viridis')


# In[56]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[57]:


embark = pd.get_dummies(train['Embarked'],drop_first=True)


# In[58]:


embark.head()


# In[59]:


train = pd.concat([train,sex,embark],axis=1)


# In[60]:


train.head()


# In[61]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[62]:


train.head()


# In[63]:


X = train.drop('Survived',axis=1)
y = train['Survived'] 


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[66]:


from sklearn.linear_model import LogisticRegression


# In[67]:


logmodel = LogisticRegression()


# In[68]:


logmodel.fit(X_train,y_train)


# In[69]:


predictions = logmodel.predict(X_test)


# In[70]:


from sklearn.metrics import classification_report


# In[71]:


print(classification_report(y_test,predictions))


# In[72]:


from sklearn.metrics import confusion_matrix


# In[73]:


confusion_matrix(y_test,predictions)

