import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# Replace 'path/to/training_dataset.csv' and 'path/to/testing_dataset.csv' with your actual file paths
train_data_path = 'Training.csv'
test_data_path = 'Testing.csv'

# Load training dataset
train_data = pd.read_csv(train_data_path)

# Load testing dataset
test_data = pd.read_csv(test_data_path)

# Drop non-numeric columns
train_data_numeric = train_data.select_dtypes(include=['number'])
test_data_numeric = test_data.select_dtypes(include=['number'])

# Separate features (X) and target variable (y) for training and testing
X_train = train_data_numeric.drop(['attack'], axis=1)
y_train = train_data_numeric['attack']

X_test = test_data_numeric.drop(['attack'], axis=1)
y_test = test_data_numeric['attack']

# Fill NaN values with 0 for both training and testing datasets
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# Initialize KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model on the entire training dataset
knn.fit(X_train, y_train)

# Make predictions on the testing set
y_test_pred = knn.predict(X_test)

# Evaluate the model on the testing set
accuracy = accuracy_score(y_test, y_test_pred)
conf_matrix = confusion_matrix(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

# Print evaluation metrics for testing
print("Testing Dataset Evaluation:")
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# ROC-AUC Curve for testing
y_test_prob = knn.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve for testing
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Testing')
plt.legend(loc='lower right')
plt.show()
