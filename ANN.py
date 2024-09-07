import pandas as pd
import numpy as np
import faiss
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('cleaned_data_before_pca.csv') #choose cleaned_data_with_pca.csv or cleaned_data_before_pca.csv

# Normalize the data by dropping the label column and scaling the features
X = data.drop(columns=['Label']).values  # Features
y = data['Label'].values  # Target (benign/attack)

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# FAISS setup for Approximate Nearest Neighbors (ANN)
dimension = X_train.shape[1]  # Number of features
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean distance)

# Add training data to the index
index.add(np.ascontiguousarray(X_train, dtype=np.float32))

# Perform ANN search for the test data
k = 5  # Number of neighbors
D, I = index.search(np.ascontiguousarray(X_test, dtype=np.float32), k)

# Get predictions (majority class among neighbors)
y_pred = np.array([np.argmax(np.bincount(y_train[I[i]])) for i in range(len(I))])

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')  # Weighted for class imbalance
recall = recall_score(y_test, y_pred, average='weighted')  # Weighted for class imbalance
f1 = f1_score(y_test, y_pred, average='weighted')  # F1 score for imbalanced data

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1-score: {f1 * 100:.2f}%')

# Insight: Analyze average distance between benign and attack traffic
benign_mask = y_test == 0  # 0 is the label for 'BENIGN'
attack_mask = y_test != 0

avg_dist_benign = np.mean(D[benign_mask])
avg_dist_attack = np.mean(D[attack_mask])

print(f'Average distance for benign traffic: {avg_dist_benign:.4f}')
print(f'Average distance for attack traffic: {avg_dist_attack:.4f}')
