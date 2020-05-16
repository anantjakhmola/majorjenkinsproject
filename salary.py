
# In[1]:
#LinearRegression just a test thing here

import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('SalaryData.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[ ]:





# In[5]:


y = dataset['Salary']


# In[6]:


x = dataset['YearsExperience']


# In[7]:


type(x)


# In[8]:


X = x.values.reshape(30,1)


# In[9]:


X.shape


# In[10]:


y.shape


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


# shift + tab


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


model = LinearRegression()

# y = b + c x


# In[16]:


#  know : x , y 

# find: c and b

# MAE : error : min error is the best
model.fit(X_train, y_train)


# In[17]:


y_pred = model.predict(X_test)


# In[18]:


y_pred


# In[19]:


y_test


# In[20]:


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')


# In[21]:


model.coef_


# In[22]:


model.intercept_


# In[23]:


# linear vwereegression : linear algebra : linear function
# y = b + cx
# y = b + 9449 x


# In[24]:


1.1*9449.96232146 + b


# In[25]:


1.5*9449.96232146


# In[26]:


# fresh , exp=0
# exp = x
# y = b + cx
# weight = c = coefficient

# fresh: initial salary offer == constant = b = bias
# y = b

# weight * 1.1


y= 25792 +  9449 * 1.1


# In[ ]:





# In[27]:


from sklearn import metrics


# In[28]:


# close to zero, was best model : ideal case
# MAE

# loss function / error 
metrics.mean_absolute_error(y_test,y_pred)


# In[29]:


# MSE : penalty

# better to use
metrics.mean_squared_error(y_test ,y_pred)


# In[30]:


# RMSE


# In[31]:


from sklearn.externals import joblib


# In[32]:


joblib.dump(model, 'salary_model.pk1')


# In[ ]:





# In[33]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[34]:


plt.scatter(X,y)


# In[ ]:




