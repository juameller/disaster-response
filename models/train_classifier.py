
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


def parse_inputs():

    """ Parses the input arguments
    
    Input: Command line inputs specified by the user.
    Output: Parsed command line inputs

    """

    parser = argparse.ArgumentParser(description='ML Pipeline')
    parser.add_argument('database_filepath', type = str, help = 'Database')
    parser.add_argument('classifier_filepath', type = str, help = 'Picle file')
    parser.add_argument('--table_name',type = str, default = 'disaster', help = 'Table name')
    parser.add_argument('--starting_verb',type = bool, default = False, help = 'Use StartingVerb Transformer')
    parser.add_argument('--keyword_search',type = bool, default = False, help = 'Use KeyWordSearch Transformer')
    parser.add_argument('--wordcount',type = bool, default = False, help = 'Use WordCount Transformer')
    


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
    Y = df.drop(['message', 'genre'], axis = 1)

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


def build_model(args):
    # Create new pipeline
    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ]))
                #('starting_verb', StartingVerbExtractor()),
                #('keyword_search',KeyWordSearch()),
                #('wordcount', WordCount())
            ])),

            ('clf', RandomForestClassifier())
        ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': [(1,1),(1,2)],
        #'features__text_pipeline__vect__max_df': (0.85, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000),
        #'clf__estimator__n_estimators': [200,500],
        #'clf__min_samples_split': [2, 3, 4],
        #'features__starting_verb__active': [args.starting_verb],
        #'features__keyword_search__active': [args.keyword_search],
        #'features__wordcount__active': [args.wordcount],
        'features__text_pipeline__tfidf__use_idf': [True, False]

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    print(classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, model_filepath):
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))





def main():
    args = parse_inputs()

    database_filepath, model_filepath, table_name = args.database_filepath, args.classifier_filepath, args.table_name
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath, table_name)



    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    
        
    print('Building model...')
    model = build_model(args)
        

    print('Training model...')
    model.fit(X_train, Y_train)
        
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

    


if __name__ == '__main__':
    main()