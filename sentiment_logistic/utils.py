import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import re
import numpy as np

def process_tweet(tweet):
    stopwords_english = set(stopwords.words('english'))
    
    # remove stock marker tickers (like $GE)
    tweet2 = re.sub(r'\$\w*', '', tweet)
    
    # remove old style retweet prefix
    tweet2 = re.sub(r'^RT\s+', '', tweet2)
    
    # remove hyperlinks
    tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
    
    # remove hash sign (#)
    tweet2 = re.sub(r'#', '', tweet2)
    
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    stemmer = PorterStemmer()
    
    tokens = tknzr.tokenize(tweet2)
    
    return [stemmer.stem(tkn) for tkn in tokens if tkn not in stopwords_english and tkn not in punctuation]


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label for each tweet
    Output:
        freqs: a dictionary mapping each (word, sentiment label) to its frequency
    
    """
    freqs = {}
    labels = np.squeeze(ys).tolist()
    for tweet, y in zip(tweets, labels):
        for token in process_tweet(tweet):
            pair = (token, y)
            freqs[pair] = freqs.get(pair, 0.) + 1
            
    return freqs
    