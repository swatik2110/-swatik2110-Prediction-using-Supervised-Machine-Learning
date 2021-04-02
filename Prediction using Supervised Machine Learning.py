#!/usr/bin/env python
# coding: utf-8

# # Author: Swati Kothmire
# #Task1: Prediction using supervised Machine Learning
# 

# # GRIP @ The Spark Foundation 
#   #In this regression task I tried to predict the percentage of marks that a student is expected to score based upon the number   of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[17]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# # Step1- Reading the data from the source

# In[18]:


# Reading data from remote link
url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)
print("Data imported successfully")

s_data.head(10)


# # Step2- Data Visualization

# In[19]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.

# # Step3 - Data processing

# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[7]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 


# In[10]:





# # Step4 - Model Training

# The next step is to split this data into training and test sets.

# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train.reshape(-1,1), y_train) 

print("Training complete.")


# In[29]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line, color='red');
plt.show()


# # Step5 - Making Predictions

# In[21]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[22]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# # Step 6 - Comparing Actual result to the Predicted Model result

# In[30]:


# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[ ]:





# # Step 7 - Evaluating the model¶

# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. Here different errors have been calculated to compare the model performance and predict the accuracy.

# In[31]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:




