# Import libraries
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

import numpy as np
# Load California housing dataset
housing = fetch_california_housing(as_frame=True)

# Create DataFrame from the dataset
df =housing.frame

print("California Housing Dataset:")
print(df.head())

# Features and target variable
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)  

# Evaluate the model using MSE and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

print("Model Coefficients:")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Coefficients for each feature
coef_df=pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])
print("Coefficients for each feature:")
print(coef_df)

# Test the model
new_data = pd.DataFrame({
    'MedInc': [8.3252],
    'HouseAge': [41.0],
    'AveRooms': [6.9841],
    'AveBedrms': [1.0238],
    'Population': [322.0],
    'AveOccup': [2.5556],
    'Latitude': [37.88],    
    'Longitude': [-122.23]
})

predicted_value = model.predict(new_data)
print(f"\nPredicted Median House Value for new data: {predicted_value[0]:,.2f}") 

print("\nEnd of California Housing Price Prediction Script.")