#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation - Data Science & Business Analytics Internship
# # Task1 - Prediction using Supervised ML
# A Linear Regression task to predict the percentage of a student based on the number of study hours per day.
# 
# DataSet Url: http://bit.ly/w-data
# 
# Batch: November 2021
# 
# # Author: Sanskar Mundra

# In[27]:


# importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[17]:


#Reading data from remote link
url="http://bit.ly/w-data"
data=pd.read_csv (url)
data.head()


# ## Plotting the distribution of scores

# In[19]:


data.plot(x="Hours",y="Scores",style='o')
plt.show()


# ## Split data in test and train

# In[23]:


y=data[["Scores"]].values
x=data[["Hours"]].values
X_train, X_test, y_train, y_test=train_test_split(x,y, test_size=0.2,random_state=0)


# ## Train our algorithm by using linear regression algorithm of Sklearn

# In[24]:


lg=LinearRegression()
lg.fit(X_train,y_train)


# ## Traning complete
# ## Plotting the regression line

# In[9]:


line = lg.coef_*x+lg.intercept_
plt.scatter(x, y)
plt.plot(x, line,color ='y');
plt.show()


# ##  Accuracy of algorithm

# In[10]:


a=lg.score(X_train,y_train)
print("Accuracy of linear Regression is = {}".format(a))


# ##  Predict score if a student studies for 9.25 hrs/days

# In[11]:


hours = [[9.25]]
predict=lg.predict(hours)
print("Score of student studies for 9.25 hrs/days is ",predict)


# ## Evaluating the model
# The final step is to evaluate the performance of the algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.

# In[12]:


print(X_test) # Testing data - In Hours
y_pred = lg.predict(X_test) # Predicting the scores


# In[13]:


print("Actual is {}".format(y_test))
print("predictis {}".format(y_pred))


# In[28]:


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R^2 Score:', metrics.r2_score(y_test, y_pred))


# ## Conclusion
# Looking at the evaluation results we can concluse that the model has good Mean Absolute Error and R^2 score
