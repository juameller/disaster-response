
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
    X = df[['message']] 
    Y = df.drop(['message'], axis = 1)

    # Obtain category names
    category_names = Y.columns

    

    # We now convert the DF to arrays:
    X = X.values
    Y = Y.values

    return X,Y, category_names
    


def tokenize(text):
    pass


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