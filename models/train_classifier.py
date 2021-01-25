
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
    try:
        engine = create_engine('sqlite:///'+database_filepath)
        df = pd.read_sql(table_name, engine)
    except:
        sys.exit("The database has not been found. Check that the file exists. Ending program..")

    
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
    """ Custom transformer to obtain the normalized number of words per message:
        
    Methods:
        - __init__ 
        - fit
        - transform
    Attributes:
        - self.active
        - self.standardscaler
        

    """

    def __init__(self, active = False):

        """
        The __init__ method instantiates a StandardScaler object. Since the X matrix will be small values, 
        we want to normalize the number of words too.
        
        Inputs:
            - active: If True, we apply the transformer, if False, the transformer does not do anything.
            This argument has been added so that we can evaluate whether the added feature increases the 
            F1 score in the gridsearch process.
        Output:
            - None
        

        """
        # Define the standardscaler object and active atribute:
        self.standardscaler = StandardScaler()
        self.active = active
        

    def fit(self, X, y=None):

        """
        Counts words of each message and fits the StandardScaler.

        Inputs: 
            - X: Array with the messages
        Output:
            - None 
        """

        # Fit the standardScaler
        if self.active:
            self.standardscaler.fit(pd.Series(X).str.split().str.len().values.reshape(-1,1).astype(np.float))
        return self

    def transform(self, X):

        """
        Obtains the normalized word count with using the StandardScaler fitted in the fit method:

        Inputs:
            -X: Array with the messages

        Outputs:
            - The transformed DF if active = True or None otherwise
        
        """
        if self.active:
            # We will always normalize the results using the mean and std obtained in the fit method:
            return pd.DataFrame(self.standardscaler.transform(pd.Series(X).str.split().str.len().values.reshape(-1,1).astype(np.float)))
        return None




class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    """ Custom transformer to asses is any of the sentences of the messages begin with a verb or RT:
        
    Methods:
        - __init__ 
        - fit
        - transform
    Attributes:
        - self.active
        
    """

    def __init__(self, active = False):

        """
        Defines whether this transformer will be used or not.
        
        Inputs:
            - active: If True, we apply the transformer, if False, the transformer does not do anything.
            This argument has been added so that we can evaluate whether the added feature increases the 
            F1 score in the gridsearch process.
        Output:
            - None
        """

        self.active = active



    def starting_verb(self, text):

        """
        This function is applied to each row to asses if any of its sentences starts with a verb or RT.
        
        Inputs:
            - text: Single message.
        Output:
            - None
        """


        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            # Since we are removing punctuation, some sentences might be completely empty.
            # In order to prevent an IndexError Exception:
            try:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
            except IndexError:
                # There was an empty sencente (due to punctuation), so we want it ignored
                pass
                
        return False


    def fit(self, X, y=None):

        """
        All estimators need a fit method. However, since in this case there is nothing to fit
        it just returns the same object

        """


        return self

    def transform(self, X):

        """
        Converts X to a dataframe and applies the function starting_verb

        Inputs:
            -X: Array with the messages

        Outputs:
            - The transformed DF if active = True or None otherwise
        
        """


        if self.active:
            X_tagged = pd.Series(X).apply(self.starting_verb)
            return pd.DataFrame(X_tagged)
        return None




class KeyWordSearch(BaseEstimator, TransformerMixin):

    """ Custom transformer to add new feaures like keywords:
        
    Methods:
        - __init__ 
        - fit
        - transform
    Attributes:
        - self.active
        
    """

    
    def __init__(self, keywords = ['medical', 'doctor', 'injection', 'sick', 'bandage', 'help', 'alone', 'child', 'water','thirst', 'drought'\
           ,'rescue', 'search','trapped','lost', 'food', 'famine','dead','death','money','homeless','accident','flood'\
           ,'fire', 'wildfire','hospital','aid','storm','hail','earthquake','cold','blanket','weather'], active = False):

        """
        Defines whether this transformer will be used or not and the keywords to add.
        
        Inputs:

            - keywords: Features to add.
            - active: If True, we apply the transformer, if False, the transformer does not do anything.
            This argument has been added so that we can evaluate whether the added feature increases the 
            F1 score in the gridsearch process.

        Output:
            - None
        """
        self.keywords = keywords
        self.active = active

    def check_keywords(self, row):

        """

        For each row (each message), it tokenize its words and returns a Series indicating if any 
        of the keywords are present in the message (similar to one-hot encoding)

        Inputs:
            - row: Row of data (message).
        Output:
            - Series indicating the presence of the keywords

        """


        wordset =  set(tokenize(row.values[0]))
        for item in self.keywords:
            row[item] = 0
            if item in wordset:
                row[item] = 1
        return row[self.keywords]



    def fit(self, X, y=None):

        """
        All estimators need a fit method. However, since in this case there is nothing to fit
        it just returns the same object

        """

        return self

    def transform(self, X):

        """
        If self.active is True, returns a data frame with the apparition of the keywords in each message.

        Inputs:
            - X: Array of messages.
        Output: 
            - The transformed DF if self.active is True (None otherwise).

        """

        if self.active:
            df = pd.DataFrame(X)
            df_tagged = df.apply(self.check_keywords, axis = 1)
            return df_tagged
        return None


def build_model(args):

    """
    This functions builds a pipeline and defines the exhaustive grid search
    to find the best estimator.

    Inputs: 
        - args: Arguments specified by the user.
    Output: 
        - cv: ML Model

    """

    # NOTE I: I HAVE COMMENTED SOME LINES SO THAT THE MODEL WOULD RUN FASTER
    # HOWEVER, THEY COULD BE UNCOMMENTED TO SEARCH OVER MORE PARAMETERS
    # OR/AND THE CUSTOM TRANSFORMERS.

    #NOTE II: If you decide to try it, be careful with the commas.

    # Create new pipeline
    pipeline = Pipeline([
            ('features', FeatureUnion([

                ('text_pipeline', Pipeline([
                    ('vect', CountVectorizer(tokenizer=tokenize)),
                    ('tfidf', TfidfTransformer())
                ])),
                #('starting_verb', StartingVerbExtractor()),
                #('keyword_search',KeyWordSearch()),
                ('wordcount', WordCount())
            ])),

            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
        #'features__text_pipeline__vect__ngram_range': [(1,1),(1,2)],
        #'features__text_pipeline__vect__max_df': (0.85, 1.0),
        #'features__text_pipeline__vect__max_features': (None, 5000),
        #'clf__estimator__n_estimators': [200,500],
        #'clf__estimator__min_samples_split': [2, 3, 4],
        #'features__starting_verb__active': [args.starting_verb],
        #'features__keyword_search__active': [args.keyword_search],
        'features__wordcount__active': [True, False],
        #'features__text_pipeline__tfidf__use_idf': [True, False]

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
    This functions evaluates the trained model and prints the precision, recall and
    f1-score for the test set.

    Inputs:
        -model: Trained model.
        - X_test: Array with the test messages.
        - Y_test: Array with the test labels.
        - category_names: List with the 36 categories

    Output:
        - None
    """

    # Make predictions
    Y_pred = model.predict(X_test)
    # Print precision, recall and accuracy for the '1' labels:
    print(classification_report(Y_test, Y_pred, target_names = category_names, zero_division = 0))


def save_model(model, model_filepath):

    """
    This functions saves the trained model into a pickle file.

    Inputs: 
        - model
        - model_filepath

    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))





def main():

    # Parse input arguments
    args = parse_inputs()

    database_filepath, model_filepath, table_name = args.database_filepath, args.classifier_filepath, args.table_name

    
    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath, table_name)


    # Split in train a test sets:
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