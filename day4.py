# Import required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,f1_score,recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset and split it into training and testing sets
data = pd.read_csv("spambase.csv")
X=data.drop('spam',axis=1)
y=data['spam']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train the Logistic regression model to classify emails as spam or not spam
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(X_test)

# Evaluate the model using accuracy,confusion matrix, precision, recall and F1-score
accuracy =  accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)

print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Precision: {precision*100:.2f}%')
print(f'Recall: {recall*100:.2f}%')
print(f'F1-score: {f1*100:.2f}%')

# Visualise  the confusion matrix using Seaborns heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.show()
