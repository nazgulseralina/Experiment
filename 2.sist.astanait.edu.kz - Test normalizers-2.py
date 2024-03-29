#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import classification_report, f1_score

# Загрузка датасета Credit Card Fraud Detection Dataset
# колонка "Class" с метками классов (0 для нормальных транзакций, 1 для мошеннических)
df = pd.read_csv('creditcard.csv')

# Разделение данных на признаки (X) и метки классов (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Применение методов нормализации данных
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(), QuantileTransformer(), Normalizer(), FunctionTransformer(), PolynomialFeatures()]

for scaler in scalers:
    # Нормализация данных
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели AdaBoost
    ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42)
    ada_boost.fit(X_train_scaled, y_train)

    # Получение предсказаний
    y_pred = ada_boost.predict(X_test_scaled)

    # Оценка качества модели с использованием F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score for {type(scaler).__name__} with AdaBoost: {f1}")
    print("\n")


# In[ ]:


Random Forest


# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import classification_report, f1_score

# Загрузка датасета Credit Card Fraud Detection Dataset
# колонка "Class" с метками классов (0 для нормальных транзакций, 1 для мошеннических)
df = pd.read_csv('creditcard.csv')

# Разделение данных на признаки (X) и метки классов (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Применение методов нормализации данных
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(), QuantileTransformer(), Normalizer(), FunctionTransformer(), PolynomialFeatures()]

for scaler in scalers:
    # Нормализация данных
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели RandomForest
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
    random_forest.fit(X_train_scaled, y_train)

    # Получение предсказаний
    y_pred = random_forest.predict(X_test_scaled)

    # Оценка качества модели с использованием F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score for {type(scaler).__name__} with RandomForest: {f1}")
    print("\n")


# KNeighborsClassifier

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import classification_report, f1_score

# Загрузка датасета Credit Card Fraud Detection Dataset
# колонка "Class" с метками классов (0 для нормальных транзакций, 1 для мошеннических)
df = pd.read_csv('creditcard.csv')

# Разделение данных на признаки (X) и метки классов (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Применение методов нормализации данных
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(), QuantileTransformer(), Normalizer(), FunctionTransformer(), PolynomialFeatures()]

for scaler in scalers:
    # Нормализация данных
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели KNeighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Получение предсказаний
    y_pred = knn.predict(X_test_scaled)

    # Оценка качества модели с использованием F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score for {type(scaler).__name__} with KNeighbors: {f1}")
    print("\n")


# Decision Tree

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import classification_report, f1_score

# Загрузка датасета Credit Card Fraud Detection Dataset
# колонка "Class" с метками классов (0 для нормальных транзакций, 1 для мошеннических)
df = pd.read_csv('creditcard.csv')

# Разделение данных на признаки (X) и метки классов (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Применение методов нормализации данных
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(), QuantileTransformer(), Normalizer(), FunctionTransformer(), PolynomialFeatures()]

for scaler in scalers:
    # Нормализация данных
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели Decision Tree
    decision_tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    decision_tree.fit(X_train_scaled, y_train)

    # Получение предсказаний
    y_pred = decision_tree.predict(X_test_scaled)

    # Оценка качества модели с использованием F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score for {type(scaler).__name__} with Decision Tree: {f1}")
    print("\n")


# In[ ]:


LogisticRegression


# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer, Normalizer, FunctionTransformer, PolynomialFeatures
from sklearn.metrics import classification_report, f1_score

# Загрузка датасета Credit Card Fraud Detection Dataset
# колонка "Class" с метками классов (0 для нормальных транзакций, 1 для мошеннических)
df = pd.read_csv('creditcard.csv')

# Разделение данных на признаки (X) и метки классов (y)
X = df.drop('Class', axis=1)
y = df['Class']

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Применение методов нормализации данных
scalers = [StandardScaler(), MinMaxScaler(), RobustScaler(), PowerTransformer(), QuantileTransformer(), Normalizer(), FunctionTransformer(), PolynomialFeatures()]

for scaler in scalers:
    # Нормализация данных
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Обучение модели Logistic Regression
    logistic_regression = LogisticRegression(random_state=42)
    logistic_regression.fit(X_train_scaled, y_train)

    # Получение предсказаний
    y_pred = logistic_regression.predict(X_test_scaled)

    # Оценка качества модели с использованием F1-score
    f1 = f1_score(y_test, y_pred)
    print(f"F1-score for {type(scaler).__name__} with Logistic Regression: {f1}")
    print("\n")

