# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#Download the required NLTK data(only needed once)
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

# Initialise the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Sample text  for emotion detection
text ="""
I am so happy today! The sun is shining and everything feels perfect. I can't wait to go out and enjoy the day. However, I am a bit nervous about the presentation I have to give later. But overall, it's a great day!
"""

# Function to detect emotiion in text
def detect_emotion(text):
    # Analyze the sentiment of the text
    sentiment_scores = sid.polarity_scores(text)
    
    # Display the sentiment scores
    print("Sentiment Scores:", sentiment_scores)

    # Determine emotion based on scores
   
    if sentiment_scores["compound"] >= 0.05:
        emotion = "Joy"
    elif sentiment_scores["compound"] <= -0.05:
        emotion = "Sadness"
    else:
        emotion = "Neutral"
    return emotion  

# Detect and print the emotion
emotion = detect_emotion(text)
print("Detected Emotion:", emotion)


