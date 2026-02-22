# Import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset of user ratings(replace with a real dataset if available)
data = {
    'user_id': [1,1,1,2, 2,3,3, 3, 4, 4],
    'book_title': ['Book A', 'Book B', 'Book C', 'Book A', 'Book D', 'Book B', 'Book C', 'Book E', 'Book A', 'Book C'],
    'rating': [5, 4, 3, 4, 5, 2, 3, 4, 5, 4]
}

# Convert to a DataFrame from the dataset
df = pd.DataFrame(data)
print("Dataset:",df)

# Create a user-book matrix
user_book_matrix = df.pivot_table(index='user_id', columns='book_title', values='rating', fill_value=0)
print("User-Book Matrix:")
print(user_book_matrix)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_book_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_book_matrix.index, columns=user_book_matrix.index)
print("User Similarity Matrix:")
print(user_similarity_df)

# Function to recommend books for a given user based on uswr similarity
def recommend_books(user_id, similarity_matrix, user_book_matrix, top_n=3):
    if user_id not in similarity_matrix.index:
        print("User not found in the dataset.")
        return []
    
    # Get the similarity scores for the given user
    similar_users = similarity_matrix[user_id].sort_values(ascending=False).drop(user_id)
    
    # Aggregate ratings from similar users ,weighted by similarity
    recommend_books = {}
    for sim_user,similarity in similar_users.items():
        rated_books = user_book_matrix.loc[sim_user]
        for book, rating in rated_books[rated_books > 0].items():
            if book not in user_book_matrix.loc[user_id] or user_book_matrix.loc[user_id, book] == 0:
                recommend_books[book] = recommend_books.get(book, 0) + similarity * rating

    # Sort the recommended books by score and return the top recommendations
    recommend_books = sorted(recommend_books.items(), key=lambda x: x[1], reverse=True)
    return [book for book, score in recommend_books[:top_n]]

# Get recommendations for a specific user
user_id = 1
recommended_books = recommend_books(user_id, user_similarity_df, user_book_matrix)
print(f"Recommended books for User {user_id}: {recommended_books}")
