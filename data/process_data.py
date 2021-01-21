import sys
import argparse

def load_data(messages_filepath, categories_filepath):
    pass


def clean_data(df):
    pass


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