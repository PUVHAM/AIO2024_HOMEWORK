import re
import nltk
nltk.download('stopwords')
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

def text_normalize(text):
    # Lowercasing
    text = text.lower()
    
    # Retweet old acronym "RT" removal
    text = re.sub(r'RT[\s]+', '',text)
    
    # Hyperlink removal
    text = re.sub(r'https?:\/\/.*[\ r\n]*', '', text)
    
    # Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    
    # Stemming
    stemmer = SnowballStemmer('english')
    words = text.split()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    
    return text

def one_hot_encoding(y, n_classes, n_samples):
    y_encoded = np.array(
        [np.zeros(n_classes) for _ in range(n_samples)]
    )
    y_encoded[np.arange(n_samples), y] = 1
    return y_encoded