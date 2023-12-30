#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df = pd.read_csv(r'C:\Users\hp\Desktop\CodSoft\IRIS.csv')    


# In[5]:


df.head()


# In[6]:


df['species'],categories =pd.factorize(df['species'])
df.head()


# In[7]:


df.describe     


# In[8]:


df.isna().sum()


# In[25]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.petal_length, df.petal_width, df.species,color='purple')
ax.set_xlabel('PetalLengthCm')
ax.set_ylabel('PetalWidthCm')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot')
plt.show()   


# In[27]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.sepal_length, df.sepal_width, df.species,color='green')
ax.set_xlabel('SepalLengthCm')
ax.set_ylabel('SepalWidthCm')
ax.set_zlabel('Species')
plt.title('3D Scatter Plot')
plt.show()


# In[43]:


sns.scatterplot(data=df, x="sepal_length", y="sepal_width",hue="species");


# In[12]:


sns.scatterplot(data=df, x="petal_length", y="petal_width",hue="species");     


# In[66]:


iris = range(1,10)
yash=[]

for k in iris:
  km = KMeans(n_clusters=k)
  km.fit(df[[ 'petal_length', 'petal_width']])
  yash.append(km.inertia_)


# In[67]:


yash


# In[45]:


plt.xlabel('k_rng')
plt.ylabel("Sum of Squared errors")
plt.plot(k_rng, sse,color='gold')     


# In[16]:


km = KMeans(n_clusters=3,random_state=0,)
y_predicted = km.fit_predict(df[['petal_length','petal_width']])
y_predicted


# In[17]:


df['cluster']=y_predicted
df.head(150)


# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(df.species, df.cluster)
cm


# In[63]:


true_labels = df.species
predicted_labels= df.cluster

cm = confusion_matrix(true_labels, predicted_labels)
class_labels = ['Setosa', 'versicolor', 'virginica']

# Plot confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.pink)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Fill matrix with values
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')

plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# In[ ]:




