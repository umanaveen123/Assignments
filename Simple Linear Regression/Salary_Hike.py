#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df=pd.read_csv("Salary_Data.csv")


# In[2]:


df.head(3)


# In[5]:


df.size


# In[3]:


df.shape


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.scatter(x=df["YearsExperience"], y=df["Salary"], color ="black")


# In[8]:


#positive relation
df.corr()


# In[9]:


#Stongly correlated
x = df[["YearsExperience"]]
y = df["Salary"]


# In[10]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()


# In[11]:


LR.fit(x,y)


# In[13]:


LR.intercept_


# In[14]:


LR.coef_


# In[15]:


y_pred=LR.predict(x)


# In[ ]:


x=df["YearsExperience"]
 y=y_pred


# In[18]:


plt.scatter(x=df["YearsExperience"], y=df["Salary"], color="black")
plt.scatter(x=df["YearsExperience"], y=y_pred, color="Red")


# In[19]:


from sklearn.metrics import mean_squared_error


# In[20]:


mse=mean_squared_error(y,y_pred)
print("Mean Squre error=",mse)


# In[21]:


import numpy as np
RMSE=np.sqrt(mse)
print("squre root of mean square error=",RMSE.round(2))


# In[23]:


a1=np.array([[11]])
LR.predict(a1)


# In[ ]:




