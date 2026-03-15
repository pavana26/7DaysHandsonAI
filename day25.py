import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the dataset
df = pd.read_csv('Telco-Customer-Churn.csv')

# 2. Data Cleaning
# Drop customerID - it's a unique string, not a feature for math
df = df.drop('customerID', axis=1)

# Fix 'TotalCharges' - it has empty strings " " that cause errors
# 'coerce' turns those spaces into NaN (Not a Number)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill the newly created NaNs with 0 or the median
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# 3. Encoding
# Using drop_first=True prevents the "Dummy Variable Trap" (multicollinearity)
# This converts ALL text columns (gender, partner, etc.) into 0s and 1s
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Define features (X) and target variable (y)
# After get_dummies, 'Churn' becomes 'Churn_Yes'
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# 5. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Feature scaling
# Now that everything is a number, the scaler will work perfectly
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Model Training & Evaluation
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
print("\nClassification Report:\n", classification_report(y_test, y_pred))