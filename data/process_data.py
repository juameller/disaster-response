import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    # Load messages and categories
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
    except FileNotFoundError:
        print('The file(s) have not been found. Please check the names of the file(s)')
        sys.exit("Terminating program..")

    # Ater inspecting the csv, we've checked taht there are not NaN

    # Merge datasets on 'id'
    df = messages.merge(categories, on = 'id')

    # We can drop the columns we don't plan on using:
    df.drop(['id', 'original', 'genre'], axis = 1, inplace=True)
    return df



def clean_data(df):

    # Ceate a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';', expand = True)

    # Obtain the names of the categories. We can do this spliting the first row:

    # Each row in the categories dataframe looks like this:
    """
    0                    related-1
    1                    request-0
    2                      offer-0
    ..............................
    33                      cold-0
    34             other_weather-0
    35             direct_report-0
    Name: 0, dtype: object
    """
    columns = categories.iloc[0,:].str.split('-').str.get(0)
    categories.columns = columns

    # Convert category values into just numbers (0/1):
    categories = categories.apply(lambda x: x.str.split('-').str.get(1))

    # Fix errors in categories. In order to do so, we will first convert
    # the categries df to int32:
    categories  = categories.astype('int32')


    # CORRECTING ERRORS:

    # We will find the columns with more than 2 values:
    columns_to_correct = list(categories.nunique()[categories.nunique()>2].index)

    for column in columns_to_correct:
        # We will assume that the categories that are set to 0 are correct.
        # Therefore we will change everything that is differnt from o to 1.
        categories[column] = categories[column].apply(lambda val: 1 if val != 0 else val)


    # drop the original categories column from df and concat the transformed one
    df.drop(['categories'], axis=1, inplace = True)
    df = pd.concat([df, categories], axis = 1)

    # Note: The column child_alone only has 1 possible value (0). In order to correct
    # this problem we should manually add some messages to the dataset.

    # Note2: Since the dataset is very unbalanced, we shoud do that for every column with
    # only a couple of values. In a real project we would do the following:

    # 1. How many messages can each person in the team generate in an hour?
    # 2. Is it feasible to add messages until each label has at least 1000 examples?

    # In case it was feasible, that would greatly improve the quality (based on precision-recall)
    # of our supervised model.

    # REMOVE DUPLICATES:
    # We will first lower case everything:
    df.message = df.message.str.lower()

    # Then we will remove the messages that are exactly identical and have been categorized equally:
    df.drop_duplicates(inplace=True)
   
    
    # That leaves us with messages that are identical but were classified differently. 
    # In this case, we could perform the OR operation between these messages.
    # After a visual inspection I've seen that this makes sense (the tags that are different
    # shoud've been included in the first place).

    # The duplicated messages are:
    duplicated_messages  = df.message[df.message.duplicated()].unique()
    

    # We will now perform an OR operation:
    for message in duplicated_messages:
        # We perform an OR operation:
        aux = df[df.message == message].drop(['message'], axis = 1).sum(axis = 0)
        # Now I have to replace everything bigger than 1 by 1:
        columns_to_fix = aux[aux>1].index
        aux.loc[columns_to_fix] = 1

        # We now assign this to the original DF:
        df.loc[df.message == message, columns] = pd.Series(message).append(aux)

    
    df.drop_duplicates(inplace=True)
    
    
    return df



    


def save_data(df, database_filename):
    pass  

def parse_inputs():
    parser = argparse.ArgumentParser(description='ETL Pipeline')
    parser.add_argument('messages_filepath', type = str, help = 'Message dataset')
    parser.add_argument('category_filepath', type = str, help = 'Categories dataset')
    parser.add_argument('database_filepath', type = str, help = 'Database')
    return parser.parse_args()

def main():
    print("--Exeuting ETL Pipeline--")
    args = parse_inputs()
   
    messages_filepath, categories_filepath, database_filepath = args.messages_filepath, \
                                                                args.category_filepath, \
                                                                args.database_filepath


    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'.format(messages_filepath, 
                                                                         categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
        
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
        
    print('Cleaned data saved to database!')
    
    

if __name__ == '__main__':
    main()