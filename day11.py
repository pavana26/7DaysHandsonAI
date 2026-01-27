# Import required libraries
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample history of stock prices
data = {
    'Date': pd.date_range(start='2026-01-01', periods=10, freq='D'),
    'Close': [150, 152, 149, 153, 155, 154, 156, 158, 157, 159]
}

#Convert the data into a DataFrame
df = pd.DataFrame(data)

df['Date'] = df['Date'].map(pd.Timestamp.toordinal)
print("Stock Price Data:")
print(df.head())

X = df[['Date']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)   

print(f'Mean Squared Error: {mse:.2f}')
print(f'R2 Score: {r2:.2f}')

plt.figure(figsize=(10,6))
plt.plot(range(len(y_test)), y_test.values, label='Actual Stock Prices', marker='o')
plt.plot(range(len(y_pred)), y_pred, label='Predicted Stock Prices', marker='x')
plt.title('Actual vs Predicted Stock Prices')      
plt.xlabel('Date(Ordinal) ') 
plt.ylabel('Stock Price ($)')
plt.legend()
plt.show()

# Test the model with new data
future_date = pd.DataFrame({
    'Date': [pd.Timestamp('2026-01-11').toordinal()]    
})
predicted_price = model.predict(future_date)
print(f'Predicted Stock Price on 2026-01-11: ${predicted_price[0]:.2f}')