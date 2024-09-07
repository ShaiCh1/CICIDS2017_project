import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-processed dataset
data = pd.read_csv('cleaned_data_before_pca.csv') #choose cleaned_data_with_pca.csv or cleaned_data_before_pca.csv

# Split the data into features and labels
X = data.drop(columns=['Label'])
y = data['Label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Calculate the scale_pos_weight to handle imbalance
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# Create and train the XGBoost model with scale_pos_weight
xgb_model = xgb.XGBClassifier(eval_metric='mlogloss', scale_pos_weight=ratio)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix - XGBoost')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance
importances = xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importance_df['importance'], y=feature_importance_df['feature'])
plt.title('Feature Importance - XGBoost')
plt.show()

print("\nTop 10 important features:\n", feature_importance_df.head(10))
