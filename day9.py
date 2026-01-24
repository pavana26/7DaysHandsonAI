# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Temperature': [30, 32, 34, 31, 29, 28, 35, 33, 30, 31],
    'Humidity': [60, 62, 64, 58, 55, 57, 65, 63, 59, 61],
    'Wind Speed': [10, 12, 8, 11, 9, 10, 13, 12, 10, 11],
    'Precipitation': [0, 0.1, 0, 0, 0.2, 0.3, 0, 0, 0.4, 0.5],
    'Next Day Temperature': [32, 31, 29, 35, 36, 34, 33, 37, 38, 39]
}

df =pd.DataFrame(data)

X=df[['Temperature','Humidity','Wind Speed','Precipitation']]
y=df['Next Day Temperature']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the Linear Regression model to predict next day's temperature
model = LinearRegression()      
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

mse =mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

plt.figure(figsize=(10,6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Temperature', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Temperature', marker='x')
plt.title('Actual vs Predicted Next Day Temperature')
plt.xlabel('Test Sample Index')
plt.ylabel('Temperature')
plt.legend()
plt.show()

new_data = pd.DataFrame({
    'Temperature': [30],
    'Humidity': [60],
    'Wind Speed': [10],
    'Precipitation': [0]
})

predicted_temp = model.predict(new_data)
print(f'Predicted Next Day Temperature : {predicted_temp[0]:.2f}C')    