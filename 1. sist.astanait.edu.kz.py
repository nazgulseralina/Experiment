#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Загрузка датасета
data = pd.read_csv('creditcard.csv')

# Предобработка данных, например, масштабирование или преобразование категориальных признаков

# Разделение на признаки и целевую переменную
X = data.drop('Class', axis=1)
y = data['Class']

# Разделение на обучающий и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели AdaBoost
model = AdaBoostClassifier()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе данных
y_pred = model.predict(X_test)

# Вычисление метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
geometric_mean = geometric_mean_score(y_test, y_pred)

# Вывод результатов
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Geometric Mean:", geometric_mean)


# In[4]:


#importing libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Загрузка датасета
data = pd.read_csv('creditcard.csv')

# Предобработка данных, например, масштабирование или преобразование категориальных признаков

# Разделение на признаки и целевую переменную
X = data.drop('Class', axis=1)
y = data['Class']

# Разделение на обучающий и тестовый наборы данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация и обучение модели AdaBoost
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Предсказание на тестовом наборе данных
y_pred = model.predict(X_test)

# Вычисление метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
geometric_mean = geometric_mean_score(y_test, y_pred)

# Вывод результатов
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Geometric Mean:", geometric_mean)


# In[5]:


# Importing libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Preprocessing data, e.g., scaling or transforming categorical features

# Splitting into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the K-Neighbors model
model = KNeighborsClassifier()
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
geometric_mean = geometric_mean_score(y_test, y_pred)

# Printing the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Geometric Mean:", geometric_mean)


# Decision Tree 

# In[6]:


# Importing libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Preprocessing data, e.g., scaling or transforming categorical features

# Splitting into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the DecisionTree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
geometric_mean = geometric_mean_score(y_test, y_pred)

# Printing the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Geometric Mean:", geometric_mean)


# LogisticRegression

# In[7]:


# Importing libraries
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Preprocessing data, e.g., scaling or transforming categorical features

# Splitting into features and target variable
X = data.drop('Class', axis=1)
y = data['Class']

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initializing and training the LogisticRegression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predicting on the testing set
y_pred = model.predict(X_test)

# Calculating metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
geometric_mean = geometric_mean_score(y_test, y_pred)

# Printing the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Geometric Mean:", geometric_mean)

