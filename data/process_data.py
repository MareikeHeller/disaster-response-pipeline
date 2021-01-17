import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories from .csv-files and merge into one dataframe.
    
    Input arguments:
    messages_filepath - Path to .csv file, where messages are stored
    categories_filepath - Path to .csv file, where categories (labels) are stored
    
    Output:
    df - Dataframe of merged messages and categories
    '''
    # load messages
    messages = pd.read_csv(messages_filepath)
    print('Loaded messages successfully.')
    
    # load categories
    categories = pd.read_csv(categories_filepath)
    print('Loaded categories successfully.')
    
    # merge dataframes
    df = messages.merge(categories, how='left', on='id')
    return df
    

def clean_data(df):
    '''
    Clean merged dataframe:
    1. Split categories into individual columns
    2. Rename individual columns according to respective category
    3. Clean label information for all categories
    4. Replace original categories column by clean individual category columns
    5. Remove duplicates
    
    Input arguments:
    df - Dataframe of merged messages and categories
    
    Output:
    df - Dataframe of merged messages and categories after cleaning
    '''
    
    # 1. create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # 2. rename column names for categories
    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # 3. clean label information for all categories
    # set each value to be the last character of the string
    # if last character > 1, set to 1
    for column in categories:
        categories[column] = categories[column].str[-1:].apply(lambda x: 1 if (int(x) > 1) else x)
        categories[column] = pd.to_numeric(categories[column])
    
    # 4. replace original column 'categories' by individual columns per category
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories], axis=1)
    
    # 5. drop duplicate rows based on 'message'
    print(str(df[df.duplicated(subset=['message'])==True].shape[0]) + ' duplicate rows.')
    print('Dropping duplicate rows...')
    df = df.drop_duplicates(subset=['message'],keep='first')
    print(str(df[df.duplicated(subset=['message'])==True].shape[0]) + ' duplicate rows.')
    
    return df


def save_data(df, database_filename):
    '''
    Save clean dataframe of merged messages and categories into database.
    
    Input arguments:
    df - Dataframe of merged messages and categories after cleaning
    database_filename - Database name
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('labeled_messages', engine, index=False, if_exists='replace')


def main():
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
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()