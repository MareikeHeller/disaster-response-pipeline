# misc
import pandas as pd
import re

#nltk
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

#scikit-learn
from sklearn.base import BaseEstimator, TransformerMixin


def tokenize(text):
    '''
    Transform text messages (documents) into normalized & lemmatized tokens.
    Normalization steps are:
    1. Lowercase
    2. Replace URLs by placeholders
    3. Replace punctuation

    Input arguments:
    text - A single text message following a disaster

    Output:
    lem_tokens - Normalized & lemmatized tokens
    '''
    # 1. normalize lowercase
    text = text.lower()

    # 2. replace URLs by placeholder
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

    # 3. normalize to digits and numbers only (replace by whitespace)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # word tokenize
    tokens = word_tokenize(text)

    # lemmatize token by token
    lemmatizer = WordNetLemmatizer()
    lem_tokens = []
    for i in tokens:
        lem_tok = lemmatizer.lemmatize(i).strip()
        lem_tokens.append(lem_tok)

    return lem_tokens


class SentenceCountExtractor(BaseEstimator, TransformerMixin):
    '''
    This class builds a customized transformer based on whether a message contains one or multiple sentences.
    The boolean information (False = one sentence, True = multiple sentences) is used as a feature in the ML model.
    '''

    def sentence_count(self, text):
        sentence_list = nltk.sent_tokenize(text)
        if len(sentence_list) > 1:
            return True
        return False

    def fit(self, X, Y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.sentence_count)
        return pd.DataFrame(X_tagged)
