# Import libraries
import numpy as np 
import pandas as pd
from surprise import SVD
from surprise import Dataset,Reader
from surprise.model_selection import train_test_split,cross_validate
from surprise import accuracy
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('ratings.csv')
df.drop('timestamp', axis=1, inplace=True)
print("Dataset preview:")
print(df.head())

# Prepare the data for the Surprise library
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Create and train the SVD model
model = SVD()
model.fit(trainset)

# Predict on the test set
predictions = model.test(testset)
# Evaluate the model
accuracy_rmse = accuracy.rmse(predictions)
print(f'RMSE: {accuracy_rmse:.4f}')

# Make predictions for a specific user and movie
user_id=196
movie_id=242

predicted_rating = model.predict(user_id, movie_id).est
print(f'Predicted rating for user {user_id} and movie {movie_id}: {predicted_rating:.2f}')

# Visualize the distribution of ratings
sns.histplot(df['rating'], bins=20, kde=True)
plt.title('Distribution of Ratings')    
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
