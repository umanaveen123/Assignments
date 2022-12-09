#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df=pd.read_csv("delivery_time.csv")
df.shape


# In[2]:


df.head(3)


# In[4]:


df.info()


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


plt.scatter(x=df["Sorting Time"],y=df["Delivery Time"], color="black")


# In[7]:


#positive correlation between Sorting time and Delivery time
df.corr()


# In[8]:


#Strongly correlated
x=df[["Sorting Time"]]
y=df["Delivery Time"]


# In[9]:


from sklearn.linear_model import LinearRegression
Lr= LinearRegression()


# In[10]:


Lr.fit(x,y)


# In[11]:


import numpy as np
Lr.intercept_


# In[12]:


Lr.coef_


# In[13]:


y_pred=Lr.predict(x)


# In[14]:


plt.scatter( x= df["Sorting Time"], y= df["Delivery Time"], color="black")
plt.scatter( x=df["Sorting Time"], y=y_pred, color="Red")


# In[15]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y,y_pred)
print("mean square error=",mse)


# In[16]:


import numpy as np
RMSE=np.sqrt(mse)
print("squre root of mean squre error=",RMSE.round(2))


# In[17]:


a1=np.array([[8]])
Lr.predict(a1)


# In[ ]:




