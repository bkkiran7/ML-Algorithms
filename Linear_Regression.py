#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


os.getcwd()


# In[3]:


data = pd.read_csv('USA_Housing.csv')


# In[4]:


data


# In[5]:


#EDA
data.head()


# In[6]:


# Checking for Null Values
data.info()


# In[7]:


# Getting the summary of Data
data.describe()


# In[8]:


#Data Preparation

#There are no null values, so there is no need of deleting or replacing the data.
#There is no necessity of having Address column/feature, so i am dropping it.

# Dropping Address Column
data.drop(['Address'],axis=1,inplace=True)


# In[9]:


data.head()


# In[10]:


# Let's plot a pair plot of all variables in our dataframe
sns.pairplot(data)


# In[11]:


# Visualise the relationship between the features and the response using scatterplots
sns.pairplot(data, x_vars=['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population'], y_vars='Price',height=7, aspect=0.7, kind='scatter')


# In[12]:


sns.heatmap(data.corr(),annot=True)


# In[13]:


data.corr().Price.sort_values(ascending=False)


# In[14]:


sns.distplot(data.Price)


# In[15]:


#Creating a Base Model

from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()

# Putting feature variable to X
X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms','Area Population']]

# Putting response variable to y
y = data['Price']


# In[16]:


X = pd.DataFrame(pre_process.fit_transform(X))


# In[17]:


X.head()


# In[18]:


y.head()


# In[19]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)


# In[20]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[21]:


# Importing RFE and LinearRegression
from sklearn.linear_model import LinearRegression


# In[22]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()


# In[23]:


# fit the model to the training data
lm.fit(X_train, y_train)


# In[24]:


# print the intercept
print(lm.intercept_)


# In[25]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# In[26]:


#From the above result we may infer that coefficient of Columns like 'Avg. Area House Age','Avg. Area Number of Rooms' and 'Avg. Area Number of Bedrooms' are influencing more as compared to other, hence we need to do scaling.


# In[27]:


# Making predictions using the model
y_pred = lm.predict(X_test)


# In[28]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[29]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[32]:


from math import sqrt

rms = sqrt(mse)
rms


# In[33]:


#From the above result we may infer that, mse is huge which shouldn't be, hence we need to improve our model.

# Actual and Predicted
c = [i for i in range(1,1501,1)] # generating index 
fig = plt.figure(figsize=(12,8))
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# In[34]:


#Also checking through Statistical Method

import statsmodels.api as sm
X_train_sm = X_train
#Unlike SKLearn, statsmodels don't automatically fit a constant, 
#so you need to use the method sm.add_constant(X) in order to add a constant. 
X_train_sm = sm.add_constant(X_train_sm)
# create a fitted model in one line
lm_1 = sm.OLS(y_train,X_train_sm).fit()

# print the coefficients
lm_1.params


# In[35]:


print(lm_1.summary())


# In[36]:


#Dropping 'Avg. Area Number of Bedrooms' Column
X.head()


# In[37]:


X.drop([3],axis=1, inplace=True)


# In[38]:


X.head()


# In[39]:


#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 ,test_size = 0.3, random_state=2)


# In[40]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[41]:


# Importing RFE and LinearRegression
from sklearn.linear_model import LinearRegression


# In[42]:


# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()


# In[43]:


# fit the model to the training data
lm.fit(X_train, y_train)


# In[44]:


# print the intercept
print(lm.intercept_)


# In[45]:


# Let's see the coefficient
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df


# In[46]:


# Making predictions using the model
y_pred = lm.predict(X_test)


# In[47]:


from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)


# In[48]:


print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)


# In[49]:


from math import sqrt

rms = sqrt(mse)
rms


# In[50]:


# Actual and Predicted
c = [i for i in range(1,1501,1)] # generating index 
fig = plt.figure(figsize=(12,8))
plt.plot(c,y_test, color="blue", linewidth=2.5, linestyle="-") #Plotting Actual
plt.plot(c,y_pred, color="red",  linewidth=2.5, linestyle="-") #Plotting predicted
fig.suptitle('Actual and Predicted', fontsize=15)              # Plot heading 
plt.xlabel('Index', fontsize=18)                               # X-label
plt.ylabel('Housing Price', fontsize=16)                       # Y-label


# In[ ]:




