#!/usr/bin/env python
# coding: utf-8

# # importing the library and DataSet

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Salary_Data')
df


# In[3]:


df.columns


# In[4]:


df.head()


# # Checking for Null Values

# In[5]:


df.isnull().sum()


# In[6]:


sns.heatmap(df.isnull())


# data set observed that no null values is present

# # Description Of Data

# predict the salary of the employee 

# In[7]:


df.dtypes


# In[8]:


df['salary'].unique()


# In[9]:


df['salary'].nunique()


# In[10]:


df['salary']=df['salary'].astype(int)


# In[11]:


df.dtypes


# # checking the distribution

# In[12]:


sns.distplot(df['salary'],kde=True)


# # Encoding the DataFrame

# In[13]:


import sklearn
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[18]:


le=LabelEncoder()

list1=['discipline','sex','rank']
for val in list1:
    df[val]=le.fit_transform(df[val].astype(str))

df


# In[19]:


df


# In[20]:


df.head()


# In[21]:


dfcor=df.corr()
dfcor


# In[22]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),annot=True,linewidths=0.1)


# dataset observed that yrs since phd and discipline are negetively correlated and yrs service and yrs since phd are positively correlated

# # Cheking the skewness

# In[23]:


df.skew()


# # checking the outliners

# In[24]:


df.dtypes


# In[25]:


df.plot(kind='box',subplots=True)


# In[26]:


sns.pairplot(df)


# separating the columns into features and target

# In[27]:


features=df.drop('salary',axis=1)
target=df['salary']


# # Scalling the data using Min-Max Scaler:

# In[28]:


from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
from sklearn.linear_model import LinearRegression
lg=LinearRegression()
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[29]:


for i in range(0,100):
    features_train,features_test,target_train,target_test=train_test_split(features,target,test_size=.3,random_state=42)
    lg.fit(features_train,target_train)
    pred_train=lg.predict(features_train)
    pred_test=lg.predict(features_test)


# In[30]:


features_train.shape


# In[31]:


target_train.shape


# In[32]:


features_test.shape


# In[33]:


target_test.shape


# In[34]:


lg.fit(features_train,target_train)


# In[35]:


lg.coef_


# In[36]:


lg.score(features_train,target_train)


# In[37]:


pred=lg.predict(features_test)
print('predicted salary:',pred)
print('actual salary',target_test)


# In[38]:


target_pred = lg.predict(features_test)


# In[39]:


c = [i for i in range (1,len(target_test)+1,1)]
plt.plot(c,target_test,color='r',linestyle='-')
plt.plot(c,target_pred,color='b',linestyle='-')
plt.xlabel('salary')
plt.ylabel('index')
plt.title('Prediction')
plt.show()


# In[40]:


print('error')
print('Mean absolute error:',mean_absolute_error(target_test,pred))
print('Mean squared erroe:',mean_squared_error(target_test,pred))
print('Root Mean Squared Error:',np.sqrt(mean_squared_error(target_test,pred)))


# In[41]:


pred_test=lg.predict(features_test)


# In[42]:


from sklearn.metrics import r2_score
print(r2_score(target_test,pred))


# In[43]:


plt.figure(figsize=(12,6))
plt.scatter(target_test,target_pred,color='r',linestyle='-')
plt.show()

