# Import necessary libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#Download the stopwords from NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Example text for summarization
text = """Natural language processing (NLP) is a field of artificial intelligence that enables computers to understand and process human language. It combines computational linguistics with machine learning, deep learning, and statistical modeling. NLP is used in various applications such as chatbots, sentiment analysis, and language translation. The goal of NLP is to read, decipher, understand, and make sense of human language in a valuable way."""   

# Function to generate a frequency-based summary
def summarize_text(text, num_sentences=2):
    # Tokenize the text into sentences and words
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())

    # Filter out stop words and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    word_freq = {}
    
    for word in words:
        if word.isalpha() and word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score sentences based on the frequency of words
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
    
    # Sort sentences by score and select the top 'num_sentences'
    summary_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = ' '.join(summary_sentences)
    
    return summary

# Generate and print the summary
summary = summarize_text(text, num_sentences=2)
print("Original text:", text)
print("\nSummary:", summary)