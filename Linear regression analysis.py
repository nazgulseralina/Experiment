#!/usr/bin/env python
# coding: utf-8

# # Linear regression analysis

# ## Aim
# 
# To build a linear regression model and select features.

# ## Tasks
# 
# - Download the data and review it.
# - Implement a variable selection cycle.
# - Implement model building using the Scikit-learn library.
# - Get model quality using MSE, MAE, R^2.
# - Compare the resulting models and draw conclusions.

# ## Method

# We have dataset of software defect prediction `jm1.csv` according to https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction.

#  ## Attribute Information:
# 	
# -      1. loc             : numeric - McCabe's line count of code
# -      2. v(g)            : numeric - McCabe "cyclomatic complexity"
# -      3. ev(g)           : numeric - McCabe "essential complexity"
# -      4. iv(g)           : numeric - McCabe "design complexity"
# -      5. n               : numeric - Halstead total operators + operands
# -      6. v               : numeric - Halstead "volume"
# -      7. l               : numeric - Halstead "program length"
# -      8. d               : numeric - Halstead "difficulty"
# -      9. i               : numeric - Halstead "intelligence"
# -     10. e               : numeric - Halstead "effort"
# -     11. b               : numeric - Halstead 
# -     12. t               : numeric - Halstead's time estimator
# -     13. lOCode          : numeric - Halstead's line count
# -     14. lOComment       : numeric - Halstead's count of lines of comments
# -     15. lOBlank         : numeric - Halstead's count of blank lines
# -     16. lOCodeAndComment: numeric
# -     17. uniq_Op         : numeric - unique operators
# -     18. uniq_Opnd       : numeric - unique operands
# -     19. total_Op        : numeric - total operators
# -     20. total_Opnd      : numeric - total operands
# -     21: branchCount     : numeric - of the flow graph
# -     22. defects         : {false,true} - module has/has not one or more 
# 
# 
# 

# Постройте модель линейной регрессии зависимости Salary от остальных параметров. 

# We implement building a linear regression model variable and look at the quality of the model. Next, on the contrary, we implement the construction of the model without the participation of each variable in turn.
# 
# At each step, we check the quality of the model using MAE, MSE, R^2. 

# In[6]:


# including libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# In[7]:


# reading the datasef
df = pd.read_csv('jm1.csv')


# In[8]:


df.head()


# In[9]:


df.columns


# In[10]:


var = ['loc', 'v(g)', 'ev(g)']


# In[11]:


# train a linear regression model with a loop for each individual variable

for i in var:
  print('-'*10, i, '-'*10)
  train_X = df[[i]]
  train_y = df[['loc']]

  model = LinearRegression() 
  model.fit(train_X, train_y)

  y_predict_train = model.predict(train_X)

  train_mse = mean_squared_error(train_y, y_predict_train)
  print("Train MSE: {}".format(train_mse))

  train_mae = mean_absolute_error(train_y, y_predict_train)
  print("Train MAE: {}".format(train_mae))

  train_r2 = r2_score(train_y, y_predict_train)
  print("Train R2: {}".format(train_r2))
  print()


# In[12]:


# train a linear regression model in a loop while eliminating each individual variable in turn

for i in var:
  var_new = var.copy()
  var_new.remove(i)

  print('-'*10, ' + '.join(var_new), '-'*10)
  train_X = df[var_new]
  train_y = df[['loc']]

  model = LinearRegression() 
  model.fit(train_X, train_y)

  y_predict_train = model.predict(train_X)

  train_mse = mean_squared_error(train_y, y_predict_train)
  print("Train MSE: {}".format(train_mse))

  train_mae = mean_absolute_error(train_y, y_predict_train)
  print("Train MAE: {}".format(train_mae))

  train_r2 = r2_score(train_y, y_predict_train)
  print("Train R2: {}".format(train_r2))
  print()


# The data shows that the best model is obtained using the loc variable, while adding the v(g) or ev(g)r variables in addition to the loc variable does not significantly improve the quality of the model. Thus, the important variable is loc, and the v(g) and ev(g) variables are not important.
# 
# These results appear to be the result of a comparison or combination of several models:
# 
# - loc: Has the smallest MSE and MAE, as well as a coefficient of determination R2 equal to 1.0. This indicates that the model fits the training data accurately.
# - v(g): Shows higher MSE and MAE values and R2 is 0.67 which means this model is more error prone compared to loc.
# - ev(g): Has even higher MSE and MAE values, and an R2 of 0.27, indicating even larger errors compared to v(g) and loc.
# - v(g) + ev(g): When the results of v(g) and ev(g) are combined, MSE and MAE decrease slightly and R2 increases slightly, but is still significantly different from the loc results.
# - loc + ev(g): When loc and ev(g) are combined, MSE and MAE remain extremely low and R2 remains at 1.0, confirming an exact fit to the training data.
# - loc + v(g): When loc and v(g) are combined, extremely low MSE and MAE values are also obtained, and R2 remains at 1.0.
# 
# From this we can conclude that the loc model has incredibly high accuracy on training data, and combining it with other models does not lead to significant improvement in results. However, it is worth noting that this is only training data, and results may vary in the real world.

# In[ ]:




