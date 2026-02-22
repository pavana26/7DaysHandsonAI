# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

# Sample dataset (replace with a real dataset if available)
data = {
    'make': ['Toyota', 'Honda', 'Ford', 'BMW', 'Audi'],
    'model': ['Corolla', 'Civic', 'Focus', '3 Series', 'A4'],
    'year': [2010, 2012, 2015, 2018, 2020],
    'mileage': [50000, 30000, 40000, 20000, 10000],
    'price': [8000, 9000, 7000, 15000, 20000]

}

# Convert to a DataFrame from the dataset
df = pd.DataFrame(data)
print("Dataset:", df)

# Convert categorical columns to numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=['make', 'model'])
print("Encoded Dataset:", df_encoded)

# Define features (x) and target variable (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model using Mean Squared Error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')