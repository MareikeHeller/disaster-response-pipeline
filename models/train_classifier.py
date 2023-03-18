# misc
import sys
sys.path.append('./')
import bz2file as bz2
import pandas as pd
import pickle
# import numpy as np
from sqlalchemy import create_engine
# import re
import pickle
from modules.utils import tokenize, SentenceCountExtractor

# nltk
import nltk
nltk.download(['punkt', 'wordnet'])
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.stem import WordNetLemmatizer

# scikit-learn
from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Load clean data from database and create separate dataframes for messages (X) and categories (Y).
    
    Input arguments:
    database_filepath - Path to database with clean data stored in the table 'labeled_messages'
    
    Output:
    X - Dataframe containing messages only
    Y - Dataframe containing categories only (labels)
    category_names - List of unique categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('labeled_messages', engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = Y.columns

    return X, Y, category_names


def build_model():
    '''
    Set up an ML pipeline consisting of transformers and a multi-output classifier.
    Hyperparameter tuning is integrated using GridSearch cross-validation.
    
    Output:
    model
    '''
    # set up pipeline for transformers and classifier
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            # customized transformer
            ('sentence_count', SentenceCountExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # define hyperparameters for tuning
    parameters = {
        'features__transformer_weights': (
            {'text_pipeline': 1},
            {'text_pipeline': 0.75},
            {'text_pipeline': 0.5},
        ),
        'clf__estimator__max_features': ['sqrt', 'log2'],
    }

    # set up model
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the ML model by predicting on the test dataset and reporting
    f1, precision, recall and accuracy scores per category.
    
    Input Arguments:
    model - ML model consisting of transformers and a multi-output classifier
    X_test - Test dataset of messages
    Y_test - Test dataset of categories (labels)
    category_names - List of unique categories (labels)
    '''
    # predict on test data
    Y_pred = model.predict(X_test)

    # report f1, precision, recall and accuracy score
    accuracy_list = []
    for col in range(len(category_names)):
        print(Y_test.columns[col], ':')
        report = classification_report(Y_test.iloc[:, col], Y_pred[:, col])
        accuracy = accuracy_score(Y_test.iloc[:, col], Y_pred[:, col])
        accuracy_list.append(accuracy)
        print(report)
        print('accuracy: ' + str("%.2f" % accuracy) + '\n')


def save_model(model, model_filepath):
    '''
    Save the ML model as a compressed .pkl-file.
    
    Input Arguments:
    model - ML model consisting of transformers and a multi-output classifier
    model_filepath - Path to save model
    '''
    with bz2.BZ2File(model_filepath, 'w') as f:
        pickle.dump(model, f)
    #pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'models/train_classifier.py ./data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()
