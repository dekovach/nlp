import nltk
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
import re

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
