# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample resume and job description data
data = {
    'resume_id':[1, 2, 3],
    'resume_text': [
        "Experienced software engineer with expertise in Python, machine learning, and data analysis.",
        "Skilled data scientist with a strong background in R, statistics, and data visualization.",
        "Project manager with experience in Agile methodologies, team leadership, and project planning."
    ]

    }
job_description= "Looking for a software engineer proficient in Python and machine learning to join our data science team."
# Convert to a DataFrame from the dataset
df = pd.DataFrame(data)
print("Resumes:", df)

# Combine job description with resumes for Tf-IDF vectorization
documents = df['resume_text'].tolist() 
documents.append(job_description)

# Initialise the TfidfVectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Calculate cosine similarity between the job description and each resume
similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

# Display similarity scores for each resume
df['similarity_score'] = similarity_scores
print(" Resume Similarity Scores:\n", df[['resume_id', 'similarity_score']])

# Identify resumes that match the job requirements (threshold can be adjusted)
threshold = 0.2
matching_resumes = df[df['similarity_score'] >= threshold]
print("Matching Resumes:\n", matching_resumes[['resume_id', 'similarity_score']])  
