#!/usr/bin/env python
# coding: utf-8

# # Challange
# Dataset name: IndexData.csv
# 
# ***Date:*** Dec/2020
# 
# ***Version:*** 8.0

# **This chanllange aims to predict Brazil Stock Schange Index (IbovIndex)**

# # 1 - Import Libs

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import plotly
import plotly.offline as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()

plt.rcParams['figure.figsize'] = (8.0, 5.0)

#setup the rows and cols dimension
pd.set_option('max_rows', 200)
pd.set_option('max_columns', 1000)

import warnings
warnings.filterwarnings('ignore')


# In[2]:


#function to display accuracy plots
def plot_accs(values, accs_train, accs_test):
    plt.plot(values, accs_train, label='train')
    plt.plot(values, accs_test, label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('Max Depth')
    plt.legend()
    #plot_accs(values, accs_train, accs_test)


# In[3]:


#function to display the heatmap with dataframe correlations
def dheatmap (dfCorr, sWidth, sHeight, sYlim):
    plt.subplots(figsize=(sWidth, sHeight)) # Set up the matplotlib figure
    maskTriu = np.triu(dfCorr)  #applying a mask for the triangle in the left side.
    s = sns.heatmap(dfCorr, mask=maskTriu, annot=True, cmap="YlGnBu", vmax=1,    center=0,square=True, linewidths=.5,cbar_kws={"shrink": .5})
    s.set_ylim(sYlim, 0.0)
    s.set_xticklabels(s.get_xticklabels(), rotation = 60)


# In[4]:


def plot_compar(ytrue, ypred):
    plt.figure(figsize=(10, 6))
    plt.plot(ytrue, label='True Values')
    plt.plot(ypred, label='Predict Values')
    plt.title("Prediction")
    plt.xlabel('Observation')
    plt.ylabel('Values')
    plt.legend()
    plt.show();


# In[5]:


import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
plotly.offline.init_notebook_mode()
pio.templates.default = "plotly_white"

def plotlyCompar(ytrue, ypred):
    # Create traces
    trace0 = go.Scatter(
        y = ytrue,    
        mode = 'lines',
        line={"color": 'orange'},  
        name = 'True values'
    )
    trace1 = go.Scatter(
        y =  ypred,
        mode = 'lines',
        line={"color": '#1f78b4' },  
        name = 'Predict values'

    )

    data = [trace0, trace1]
    py.iplot(data, filename='scatter-mode')


# # 2- Load Dataset
# 
# dataset name: IndexData.csv

# In[6]:


path = os.environ['USERPROFILE']
path


# In[14]:


#to use sep=';' to load the dataset
 
dfIndexData = pd.read_csv(path + '\\CitiDSPython\\data\\IndexData.csv', sep=';')


# In[15]:


#check the first rows

dfIndexData.head()


# # 3 -  Check shape ,  info, last lines

# In[16]:


dfIndexData.shape


# In[26]:


dfIndexData.info()


# In[18]:


dfIndexData.tail()


# # 4 -  Check Nulls / NaN

# In[19]:


dfIndexData.isnull().sum()

# dfIndex.isnull().sum() / len(dfIndex) * 100
# esse mostra em porcentagem 


# # 5 -  Fix Dataset regarding Null, format values, etc

# In[22]:


dfIndexData['SPXIndex'].fillna(method='backfill', inplace = True)
dfIndexData['USDBRL'].fillna(method='backfill', inplace = True)


# In[25]:


dfIndexData['Date'] = pd.to_datetime(dfIndexData['Date'])


# In[24]:


dfIndexData['CRBCMDTIndex'] = dfIndexData['CRBCMDTIndex'].str.replace(',', '.').astype(float)
dfIndexData.head()


# # 6 - Describe View

# In[23]:


#use .describe()

dfIndexData.describe()


# # 7 - DataSet Plots 
# 
# Check the correlation
#  

# In[29]:


#you can use the useful function

dfIndexData.corr()


# In[32]:




#setup the function with dataset correlation

dheatmap(dfIndexData.corr(), sWidth = 6, sHeight = 6, sYlim = 6)


# # 8 - What do you like to Predict? Which Algo to use to predict?
# 
# Just think about it

# # 9 - Choose the Target value (y) and Predictors values (X)

# In[33]:


y = dfIndexData['IbovIndex'].values
y[:3]

X = dfIndexData.drop(['Date' , 'IbovIndex'], axis = 1)
X[:3]

# X= df[['SPXIndex', 'USDBRL','Pre_GEBR5YIndex', 'CRBCMDTIndex']]
# Essa só deixa as que queremos ver, ao invés de dropar as que não queremos


# In[34]:


dfIndexData.columns


# # 10 - Split the Dataset

# In[36]:


# ???, ???, ???, ??? = train_test_split(???, ???, test_size=???, random_state=???)

# from sklearn.model_selection import train_test_split

Xtrain , Xtest , ytrain , ytest = train_test_split( X , y , test_size = 0.3, random_state = 1)


# In[37]:


print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)


# # 11 - Choose and load the Machine Learning model 

# In[39]:


#Random Forest

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 200)
rfr


# # 12 - Fit the model

# In[41]:


#Training  the Xtrain and ytrain
rfr.fit( Xtrain, ytrain )


# # 13 -  Do the Predictions

# In[53]:



#Predictions Xtest
y_prediction = rfr.predict( Xtest )


# # 14 - Plot and display the Precictions against the True Values

# In[54]:


plotlyCompar(ytest, y_prediction)


# In[ ]:





# In[ ]:





# # 15 - Check the Metrics

# In[55]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(ytest, y_predicton))  
print('Mean Squared Error:', metrics.mean_squared_error(ytest, y_predicton))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(ytest, y_predicton))) 
print('Accuracy of Random Forest Regression on training set: {:.2f}'.format(rfr.score(Xtrain, ytrain) * 100) ,'\nAccuracy of Random Forest Regression on test set: {:.2f}'.format(rfr.score(Xtest, ytest) * 100))

