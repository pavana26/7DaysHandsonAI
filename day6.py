# import necessary libraries
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random


# Download required NLTK data files
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess the dataset and extract features
def extract_features(words):
    return {word: True for word in words }

# Load the movie reviews dataset from NLTK
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Shuffle the documents to ensure random distribution
random.shuffle(documents)

# Prepare the dataset for training and testing
featuresets = [(extract_features(doc), category) for (doc, category) in documents]
train_set, test_set = featuresets[:1600], featuresets[1600:]    

# Train the Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier on the test set
accuracy = nltk_accuracy(classifier, test_set)
print(f'Accuracy: {accuracy*100:.2f}%')

# Show the most informative features
classifier.show_most_informative_features(10)   


# Test on new input sentences
def analyse_sentiment(text):
    # Tokenize the input text and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]

    # Predict sentiment
    features = extract_features(words)
    return classifier.classify(features)

# Example usage
input_text = ["I absolutely loved this movie! The plot was thrilling and the characters were well-developed.",
    "I hated this movie.It was a absolute waste of time.",
"The plot was a bit dull,but the performance were decent.",
"I have mixed feeling about this film.It wa sokay,not great but not terrible either."
]

for sentence in input_text:
    sentiment = analyse_sentiment(sentence)
    print(f'Text: {sentence}\nSentiment: {sentiment}\n')
    print()