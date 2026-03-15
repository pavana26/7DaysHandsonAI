# Importing libraries
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df =pd.read_csv('heart_disease.csv')

print("Dataset preview:")
print(df.head())

# Preprocess the data
# Check for missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))   
X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
y = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the Logistic Regression model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Predict on the test set
y_pred_logistic = logistic_model.predict(X_test)
# Evaluate the Logistic Regression model
accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
print(f'Logistic Regression Accuracy: {accuracy_logistic * 100:.2f}%')

# Create and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Predict on the test set
rf_preds = rf_model.predict(X_test)

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, rf_preds)
print(f'Random Forest Accuracy: {accuracy_rf * 100:.2f}%')

# Evaluate the best model
best_model = rf_model if accuracy_rf > accuracy_logistic else logistic_model
best_preds = rf_preds if accuracy_rf > accuracy_logistic else y_pred_logistic

print("Best Model Metrics:")
print(f"Accuracy: {accuracy_score(y_test, best_preds) * 100:.2f}%")

print("Confusion Matrix:\n", confusion_matrix(y_test, best_preds))
print("Classification Report:\n", classification_report(y_test, best_preds))

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()  

# Make Predictions on New Data
new_data = pd.DataFrame({
    'age': [63],
    'sex': [1],
    'cp': [3],
    'trestbps': [145],
    'chol': [233],
    'fbs': [1],
    'restecg': [0],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [2.3],
    'slope': [0],
    'ca': [0],
    'thal': [1]
})


# Scale new data
new_data_scaled = scaler.transform(new_data)
prediction = best_model.predict(new_data_scaled)
print(f'Predicted class for new data: {"Disease" if prediction[0] == 1 else "No Disease"}')