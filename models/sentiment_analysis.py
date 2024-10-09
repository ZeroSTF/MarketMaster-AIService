import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def analyze(text):
    preprocessed_text = preprocess_text(text)
    sentiment_scores = sia.polarity_scores(preprocessed_text)
    
    if sentiment_scores['compound'] >= 0.05:
        sentiment = 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return {
        'original_text': text,
        'preprocessed_text': preprocessed_text,
        'sentiment': sentiment,
        'sentiment_scores': sentiment_scores
    }