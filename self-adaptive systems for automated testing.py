#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torch torchvision


# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data for the example
np.random.seed(42)
num_samples = 1000
num_features = 10

X = np.random.rand(num_samples, num_features)
y = (X.sum(axis=1) > 5).astype(int)  # Binary target based on sum of features

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define a simple neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Instantiate the network, define loss function and optimizer
net = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    net.train()
    optimizer.zero_grad()
    outputs = net(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
net.eval()
with torch.no_grad():
    outputs = net(X_test)
    predicted = outputs.round()
    accuracy = (predicted.eq(y_test).sum().item()) / y_test.size(0)
    print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
torch.save(net.state_dict(), 'self_adaptive_model.pth')

print("Model saved as 'self_adaptive_model.pth'")


# In[4]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define the neural network architecture (this should match the architecture used in training)
class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# Load the model
input_size = 10  # Adjusting to the correct input size used during training
model = SimpleNN(input_size)
model.load_state_dict(torch.load('self_adaptive_model.pth'))
model.eval()

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')

# Handle missing values if there are any
df.fillna(0, inplace=True)

# Select only the first 10 principal components (or features) for consistency with the trained model
X = df.iloc[:, :10]  # Adjust this line based on which features were used during training
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Make predictions
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    y_pred_class = (y_pred > 0.5).float()

# Convert predictions to numpy arrays for evaluation
y_pred_class = y_pred_class.numpy()
y_test = y_test_tensor.numpy()

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_class)
classification_rep = classification_report(y_test, y_pred_class)
conf_matrix = confusion_matrix(y_test, y_pred_class)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)


# In[5]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Define the neural network architecture
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')
df.fillna(0, inplace=True)

# Select the features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = AdvancedNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'advanced_self_adaptive_model.pth')
print("Model saved as 'advanced_self_adaptive_model.pth'")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    y_pred_class = (y_pred > 0.5).float()

y_pred_class = y_pred_class.numpy()
y_test = y_test_tensor.numpy()

accuracy = accuracy_score(y_test, y_pred_class)
classification_rep = classification_report(y_test, y_pred_class)
conf_matrix = confusion_matrix(y_test, y_pred_class)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)


# In[6]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Define the neural network architecture
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')
df.fillna(0, inplace=True)

# Select the features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = AdvancedNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'advanced_self_adaptive_model.pth')
print("Model saved as 'advanced_self_adaptive_model.pth'")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze()
    y_pred_class = (y_pred > 0.5).float()

y_pred_class = y_pred_class.numpy()
y_test = y_test_tensor.numpy()

accuracy = accuracy_score(y_test, y_pred_class)
classification_rep = classification_report(y_test, y_pred_class)
conf_matrix = confusion_matrix(y_test, y_pred_class)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_rep)
print("Confusion Matrix:\n", conf_matrix)


# In[7]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Define the neural network architecture
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')
df.fillna(0, inplace=True)

# Select the features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]
model = AdvancedNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = y_pred.round().squeeze()

    accuracy = accuracy_score(y_test_tensor, y_pred_class)
    class_report = classification_report(y_test_tensor, y_pred_class)
    conf_matrix = confusion_matrix(y_test_tensor, y_pred_class)
    
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(class_report)
    print('Confusion Matrix:')
    print(conf_matrix)


# In[8]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Define the neural network architecture
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')
df.fillna(0, inplace=True)

# Select the features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize and train the Isolation Forest model
iso_forest = IsolationForest(random_state=42, contamination=0.01)
iso_forest.fit(X_train)
iso_forest_pred = iso_forest.predict(X_test)

# Initialize and train the AdaBoost model
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_boost.fit(X_train, y_train)
ada_boost_pred = ada_boost.predict(X_test)

# Initialize and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_pred = decision_tree.predict(X_test)

# Neural Network Training
input_size = X_train.shape[1]
model = AdvancedNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the neural network model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the neural network model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = y_pred.round().squeeze()

    accuracy = accuracy_score(y_test_tensor, y_pred_class)
    class_report = classification_report(y_test_tensor, y_pred_class)
    conf_matrix = confusion_matrix(y_test_tensor, y_pred_class)
    
    print(f'Neural Network Accuracy: {accuracy:.4f}')
    print('Neural Network Classification Report:')
    print(class_report)
    print('Neural Network Confusion Matrix:')
    print(conf_matrix)

# Evaluate Isolation Forest model
iso_forest_accuracy = accuracy_score(y_test, iso_forest_pred)
print(f'Isolation Forest Accuracy: {iso_forest_accuracy:.4f}')

# Evaluate AdaBoost model
ada_boost_accuracy = accuracy_score(y_test, ada_boost_pred)
print(f'AdaBoost Accuracy: {ada_boost_accuracy:.4f}')

# Evaluate Decision Tree model
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
print(f'Decision Tree Accuracy: {decision_tree_accuracy:.4f}')


# In[11]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# Define the neural network architecture
class AdvancedNN(nn.Module):
    def __init__(self, input_size):
        super(AdvancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')
df.fillna(0, inplace=True)

# Select the features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Initialize and train the Isolation Forest model
iso_forest = IsolationForest(random_state=42, contamination=0.01)
iso_forest.fit(X_train)
iso_forest_pred = iso_forest.predict(X_test)

# Initialize and train the AdaBoost model with SAMME algorithm
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME')
ada_boost.fit(X_train, y_train)
ada_boost_pred = ada_boost.predict(X_test)

# Initialize and train the Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)
decision_tree_pred = decision_tree.predict(X_test)

# Neural Network Training
input_size = X_train.shape[1]
model = AdvancedNN(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the neural network model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 2 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating the neural network model
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_class = y_pred.round().squeeze()

    accuracy = accuracy_score(y_test_tensor, y_pred_class)
    class_report = classification_report(y_test_tensor, y_pred_class)
    conf_matrix = confusion_matrix(y_test_tensor, y_pred_class)
    
    print(f'Neural Network Accuracy: {accuracy:.4f}')
    print('Neural Network Classification Report:')
    print(class_report)
    print('Neural Network Confusion Matrix:')
    print(conf_matrix)

# Evaluate Isolation Forest model
iso_forest_accuracy = accuracy_score(y_test, iso_forest_pred)
print(f'Isolation Forest Accuracy: {iso_forest_accuracy:.4f}')

# Evaluate AdaBoost model
ada_boost_accuracy = accuracy_score(y_test, ada_boost_pred)
print(f'AdaBoost Accuracy: {ada_boost_accuracy:.4f}')

# Evaluate Decision Tree model
decision_tree_accuracy = accuracy_score(y_test, decision_tree_pred)
print(f'Decision Tree Accuracy: {decision_tree_accuracy:.4f}')


# In[ ]:




