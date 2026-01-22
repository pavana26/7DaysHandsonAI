# import necessary libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer 

data = {'movie_id': [1, 2, 3, 4, 5],
        'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Pulp Fiction', 'Forrest Gump'],
        'genres': ['Drama', 'Crime, Drama', 'Action, Crime, Drama', 'Crime, Drama', 'Drama, Romance']}

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Movie Dataset:")
print(df)

# Define a TF-IDF Vectorizer to convert the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the genres column into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['genres'])

# Compute the cosine similarity matrix based on the TF-IDF features
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)  

# Function to recommend movies based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = df.index[df['title'] == title].tolist()[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 3 most similar movies (excluding itself)
    sim_scores = sim_scores[1:4]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 3 most similar movies
    return df['title'].iloc[movie_indices]



# Test the recommendation system with example
input_movie = 'The Godfather'
recommendaed_movies = get_recommendations(input_movie)
print(f"\n Movies similar to '{input_movie}':")
for movie in recommendaed_movies:
    print(movie)