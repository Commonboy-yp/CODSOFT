#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#IMPORTING DATASET

df = pd.read_csv(r'C:\Users\hp\Desktop\CodSoft\titanic.csv')
df.head(10)



# In[4]:


df.shape


# In[5]:


df.describe()


# In[7]:


df['Survived'].value_counts()


# In[39]:


sns.countplot(x=df['Survived'], hue=df['Pclass'],data=df,color="green")


# In[29]:


df["Sex"]


# In[38]:


sns.countplot(x=df['Sex'], hue=df['Survived'],data=df,color="green")


# In[11]:


df.groupby('Sex')[['Survived']].mean()


# In[12]:


df['Sex'].unique()


# In[13]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

df['Sex']= labelencoder.fit_transform(df['Sex'])

df.head()


# In[14]:


df['Sex'], df['Survived']


# In[37]:


sns.countplot(x=df['Sex'], hue=df["Survived"],data=df,color="green")


# In[16]:


df.isna().sum()


# In[17]:


df=df.drop(['Age'], axis=1)


# In[18]:


df_final = df
df_final.head(10)


# In[19]:


X= df[['Pclass', 'Sex']]
Y=df['Survived']


# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[21]:


from sklearn.linear_model import LogisticRegression

log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)


# In[22]:


pred = print(log.predict(X_test))


# In[23]:


print(Y_test)


# In[48]:


import warnings
warnings.filterwarnings("ignore")

res= log.predict([[2,0]])

if(res==0):
  print("Not Survived, better luck next time")
else:
  print("Survived, hope u live well after this")


# In[ ]:





# In[ ]:




