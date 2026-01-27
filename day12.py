# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset Pima Indians Diabetes Dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=column_names)
print("Dataset Preview:")
print(data.head())

X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Logistic Regression model to predict diabetes
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  

# Evaluate the model using accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)   
print(f'Accuracy: {accuracy * 100:.2f}%')
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Visualize the confusion matrix using Seaborn's heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')   
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

new_data = pd.DataFrame({
    'Pregnancies': [5], 
    'Glucose': [120],
    'BloodPressure': [72],
    'SkinThickness': [35],
    'Insulin': [80],
    'BMI': [32.0],
    'DiabetesPedigreeFunction': [0.5],
    'Age': [42]
})

predicted_outcome = model.predict(new_data)
outcome_label = 'Diabetic' if predicted_outcome[0] == 1 else 'Non-Diabetic'
print(f'Predicted Outcome for the new data: {outcome_label}')   