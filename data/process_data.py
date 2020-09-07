import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    messages_filepath [string] = path to messages.csv file
    categories_filepath [string] = path to categories.csv file
    Function to load messages.csv and categories.csv and merge them into a dataframe.
    df [dataframe] = merged dataframe of messages.csv and categories.csv
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    '''
    df [dataframe] = merged dataframe of messages.csv and categories.csv with 'categories' column
    Function to create individual category columns and drop duplicates.
    df [dataframe] = dataframe of messages and individual category columns
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x.split('-')[0]).values.tolist()    
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)   
    # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filepath):
    '''
    df [dataframe] = cleaned dataframe
    database_filename [string] = sql database name, ex. 'data/DisasterResponse.db'
    Function to save dataframe into a sql database.
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('Disasters', engine, index=False)



def main():
    '''
    Function to parse arguments, load data, clean dataframe, and save it into a database.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python data/process_data.py '\
              'data/disaster_messages.csv data/disaster_categories.csv '\
              'data/DisasterResponse.db')


if __name__ == '__main__':
    main()