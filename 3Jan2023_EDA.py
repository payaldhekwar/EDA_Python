#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[2]:


titanic_data=pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-1/master/Data/Casestudy/titanic_train.csv')


# In[3]:


titanic_data.head()


# In[4]:


titanic_test=pd.read_csv('https://raw.githubusercontent.com/insaid2018/Term-2/master/Data/test.csv')


# In[5]:


titanic_test.head()


# In[6]:


titanic_data.shape


# In[7]:


titanic_data.tail()


# In[8]:


titanic_test.tail()


# In[9]:


combine


# In[10]:


titanic_data.describe() # only work for numerical col


# In[11]:


##insights:#total samples are 891


# In[12]:


pip install pandas_profiling


# In[13]:


import pandas_profiling 


# In[14]:


from pandas_profiling  import ProfileReport


# In[ ]:





# In[15]:


df=pd.DataFrame(np.random.rand(100,5),columns=["a","b","c","d","e"])


# In[16]:


titanic_profile=ProfileReport(titanic_data,title="Pandas Profiling Report")


# In[17]:


titanic_profile


# In[18]:


titanic_profile.to_file(output_file="Titanic_before_preprocessing.html")


# ## Data Preprocessing

# In[19]:


miss1=titanic_data.isnull().sum()
miss=(titanic_data.isnull().sum()/len(titanic_data))*100


# In[20]:


miss1


# In[21]:


miss


# In[22]:


miss_data=pd.concat([miss1,miss],axis=1,keys=['Total','%'])


# In[23]:


miss_data


# In[24]:


new_age=titanic_data.Age.median()  #wherever there is not mention age we take median


# In[25]:


new_age


# In[26]:


titanic_data.Age.fillna(new_age,inplace=True)


# In[27]:


titanic_data


# In[28]:


titanic_data.isnull().sum()


# In[29]:


titanic_data.Embarked=titanic_data.Embarked.fillna(titanic_data['Embarked'].mode()[0])


# In[30]:


titanic_data.isnull().sum()


# In[31]:


titanic_data.drop('Cabin',axis=1,inplace=True)


# In[32]:


titanic_data.isnull().sum()


# In[33]:


##Passangerid cam be drop as it is just id like a clabin


# In[34]:


titanic_data.drop('PassengerId',axis=1,inplace=True)


# In[35]:


titanic_data.isnull().sum()


# In[36]:


titanic_data.drop('Ticket',axis=1,inplace=True)


# In[37]:


titanic_data.isnull().sum()


# In[38]:


#Creating new field


# In[39]:


titanic_data['Age_band']=0


# In[43]:


titanic_data.loc[titanic_data['Age']<=1,'Age_band']="Infant"
titanic_data.loc[(titanic_data['Age']>1&(titanic_data['Age']<=12),'Age_band')]="Children"
titanic_data.loc[titanic_data['Age']>12,'Age_band']="Adult"


# In[45]:


titanic_data.head(4)


# In[47]:


titanic_data['FareBand']=0
titanic_data.loc[(titanic_data['Fare']>=0)&(titanic_data['Fare']<=10),['FareBand']]=1
titanic_data.loc[(titanic_data['Fare']>10)&(titanic_data['Fare']<=15),['FareBand']]=2
titanic_data.loc[(titanic_data['Fare']>15)&(titanic_data['Fare']<=35),['FareBand']]=3
titanic_data.loc[titanic_data['Fare']>35,['FareBand']]=4
titanic_data.head(4)


# In[48]:


for dataset in titanic_data:
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
    
pd.crosstab(titanic_data['Title'],titanic_data['Sex'])    


# In[ ]:




