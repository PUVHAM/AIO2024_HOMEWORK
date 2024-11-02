import re
import numpy as np

from nltk.tokenize import TweetTokenizer
from collections import defaultdict

def text_normalize(text):
    # Retweet old acronym "RT" removal
    text = re.sub(r'RT[\s]+', '',text)
    
    # Hyperlink removal
    text = re.sub(r'https?:\/\/.*[\ r\n]*', '', text)
    
    # Hashtags removal
    text = str.replace(r'#', '', text)
    
    # Punctuation removal
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenization
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )
    text_tokens = tokenizer.tokenize(text)
    
    return text_tokens

def get_freqs(df):
    freqs = defaultdict(lambda: 0)
    for idx, row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        
        tokens = text_normalize(tweet)
        for token in tokens:
            pair = (token, label)
            freqs[pair] += 1
            
    return freqs

def get_feature(text, freqs):
    tokens = text_normalize(text)
    
    X = np.zeros(3)
    X[0] = 1
    
    for token in tokens:
        X[1] += freqs[(token, 0)]
        X[2] += freqs[(token, 1)]
        
    return X

def run_preprocess(df):
    X = []
    y = []
    
    freqs = get_freqs(df)
    for _, row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        
        x_i = get_feature(tweet, freqs)
        
        X.append(x_i)
        y.append(label)
        
    return np.array(X), np.array(y)