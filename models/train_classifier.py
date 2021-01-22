
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


def parse_inputs():

    """ Parses the input arguments
    
    Input: Command line inputs specified by the user.
    Output: Parsed command line inputs

    """

    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('database_filepath', type = str, help = 'Database')
    parser.add_argument('classifier_filepath', type = str, help = 'Picle file')
    parser.add_argument('--table_name',type = str, default = 'disaster', help = 'Table name')

    return parser.parse_args()


def load_data(database_filepath, table_name):
    """ Loads data fromo database
    Inputs: 
        Database_filepath, table_name

    Outputs: 
        X, Y, category_names

    """

    # Load df from SQLite db:
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql(table_name, engine)
    X = df.message.values 
    Y = df.drop(['message'], axis = 1)

    # Obtain category names
    category_names = Y.columns

    # We now convert the DF to arrays:

    Y = Y.values

    return X,Y, category_names
    


def tokenize(text):
    """ Tokenize function
    Input: 
        text: single message

    Output:
        clean_tokens

    The tokenize functions performs the following tasks:

        1. Replace every URL by a placeholder.
        2. Eliminate punctuation.
        3. Replace numbers by placeholders.
        4. Tokenize text.
        5. Remove stop words.
        6. Lower case each token and lemmatize it.

    """
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # We can get rid of the puntuation:
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # We will repalce all numbers for a placeholder:
    numbers = re.findall(r'[0-9]+', text)
    for number in numbers:
        text = text.replace(number, 'nphdr')
    
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Eliminate stop words in Englisg
    words = [token for token in tokens if token not in stopwords.words('english') ]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


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

    def __init__(self):
        # Initialize the standardscaler
        self.standardscaler = StandardScaler()
        

    def fit(self, X, y=None):
        # Fit the standardScaler
        self.standardscaler.fit(pd.Series(X).str.split().str.len().values.reshape(-1,1).astype(np.float))
        return self

    def transform(self, X):
        # We will always normalize the results using the mean and std obtained in the fit method:
        return pd.DataFrame(self.standardscaler.transform(pd.Series(X).str.split().str.len().values.reshape(-1,1).astype(np.float)))


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass





def main():
    args = parse_inputs()
    database_filepath, model_filepath, table_name = args.database_filepath, args.classifier_filepath, args.table_name
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath, table_name)



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
    print('Building model...')
    model = build_model()
        
    print('Training model...')
    model.fit(X_train, Y_train)
        
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

    


if __name__ == '__main__':
    main()