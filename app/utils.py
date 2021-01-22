import sys
import argparse
import re
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet','stopwords','averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

class WordCount(BaseEstimator, TransformerMixin):
    """ We define a WordCount transformer. A custom transformer inherits from BaseEstimator and TransformerMixin
        
        def __init__(self): 
            We have to initialize StandardScaler(). Since the X matrix will be small values, we want to normalize 
            the number of words too.

        def fit(self, X, y=None): 
            We split the words, obtain the length of the resulting list and fit the StandardScaler.

        def transform(self, X):
            We obtain the normalized word count.

    """

    def __init__(self, active = False):
        # Initialize the standardscaler and wether we plan to use it or no:
        self.standardscaler = StandardScaler()
        self.active = active
        

    def fit(self, X, y=None):
        # Fit the standardScaler
        if self.active:
            self.standardscaler.fit(pd.Series(X).str.split().str.len().values.reshape(-1,1).astype(np.float))
        return self

    def transform(self, X):
        if self.active:
            # We will always normalize the results using the mean and std obtained in the fit method:
            return pd.DataFrame(self.standardscaler.transform(pd.Series(X).str.split().str.len().values.reshape(-1,1).astype(np.float)))
        return None




class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    """ We define a StartingVerb transformer. A custom transformer inherits from BaseEstimator and TransformerMixin
        
        def __init__(self): 
            We define a special case so we can skip it.

        def fit(self, X, y=None): 
            We do not have to fit anything.

        def transform(self, X):
            We convert X to a dataframe and apply the function starting_verb. Returns a Dataframe with
            the results.

        def staring_verb(self, text):
            We go through the sentences of each messages, tokenize them, obtain the part of speech tag
            and if the first word of any of the sentences of the message is either a verb or RT we 
            return True (otherwise, return false).
    """
    def __init__(self, active = False):
        self.active = active

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # Since we are removing punctuation, some sentences might be completely empty.
            # In order to prevent an IndexError:
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except IndexError:
                pass
                # There was an empty sencente (due to punctuation, we want it ignored)
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.active:
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged)
        return None




class KeyWordSearch(BaseEstimator, TransformerMixin):

    """ We define a KeyWordSearch transformer. A custom transformer inherits from BaseEstimator and TransformerMixin
        
        def __init__(self): 
            Define the Keywords we want to be searched

        def fit(self, X, y=None): 
            We do not have to fit anything.

        def transform(self, X):
            We convert X to a dataframe and apply the function check_keywords. Returns a Dataframe with
            with whether the keywords are present in the message.

        def check_keywords(self, text):
            We go through the sentences of each messages, tokenize them, obtain the part of speech tag
            and if the first word of any of the sentences of the message is either a verb or RT we 
            return True (otherwise, return false).
    """
    
    def __init__(self, keywords = ['medical', 'doctor', 'injections', 'sick', 'bandage', 'help', 'alone', 'child', 'water','thirst', 'drought'\
           ,'rescue', 'search','trapped','lost', 'food', 'famine','dead','death','money','homeless','accident','floods'\
           ,'fire', 'wildfire','hospital','aid','storm','hail','earthquake','cold','blanket','weather'], active = False):
        self.keywords = keywords
        self.active = active

    def check_keywords(self, row):
        wordset =  set(tokenize(row.values[0]))
        for item in self.keywords:
            row[item] = 0
            if item in wordset:
                row[item] = 1
        return row[self.keywords]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.active:
            df = pd.DataFrame(X)
            df_tagged = df.apply(self.check_keywords, axis = 1)
            return df_tagged
        return None