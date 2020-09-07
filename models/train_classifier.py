import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support
import pickle
nltk.download(['punkt','stopwords','wordnet'])


def load_data(database_filename):
    '''
    database_filename [string] = sql database name, ex. 'data/DisasterResponse.db'
    Function to load data from sql database.
    X [dataframe] = messages column
    y [dataframe] = category columns
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = pd.read_sql_table('Disasters', engine)
    X = df['message']
    y = df.iloc[:, 4:] # not including 'id', 'message', 'original', 'genre'
    return X, y                       


def tokenize(text):
    '''
    text [string] = sentence text
    Function to lowercase, tokenize, remove punctuations and stopwords, and lemmatize any string.
    lemmas [list of strings] = list of lemmas
    '''
    #normalize text
    text = re.sub(r'[^a-zA-Z0-9]',' ',text.lower())
    #tokenize messages
    words = word_tokenize(text)
    # remove stopwords
    tokens = [w for w in words if w not in stopwords.words("english")]
    # Lemmatization
    lemmas = [WordNetLemmatizer().lemmatize(t) for t in tokens]
    return lemmas


def build_model():
    '''
    Function to find the best parameters for an adaboost classifier model from a grid search object.
    cv [grid search] = the adaboost classifier model pipeline within a grid search object
    '''
    # create a pipeline
    pipeline = Pipeline([
        ('vect',TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
#         'vect__use_idf': (True, False),
#         'clf__estimator__n_estimators': [40, 50, 60],
#         'clf__estimator__learning_rate': [0.5, 1]
        
        'vect__use_idf': [True, False],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__learning_rate': [1]
    }
    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test):
    '''
    model [pipeline] = adaboost classifier model
    X_test [dataframe] = messages column
    Y_test [dataframe] = category columns
    Function to evaluate model predictions to true labels and calculate the average precision, recall, & f-score.
    '''
    y_pred = model.predict(X_test)
    score_df = pd.DataFrame(columns=['Label', 'Precision', 'Recall', 'Fscore'])
    with open('log.txt', 'a') as log:
        for i, col in enumerate(Y_test.columns):
            precision, recall, f_score, support = precision_recall_fscore_support(Y_test[col], y_pred[:, i], average='weighted')
            score_df.loc[i, 'Label'] = col
            score_df.loc[i, 'Precision'] = precision
            score_df.loc[i, 'Recall'] = recall
            score_df.loc[i, 'Fscore'] = f_score
        
        avg_scores = 'Average F Score: {}\n'.format(score_df['Fscore'].mean())+\
        'Average Precision: {}\n'.format(score_df['Precision'].mean())+\
        'Average Recall: {}\n'.format(score_df['Recall'].mean())
        log.write(avg_scores)
        print(avg_scores)


def save_model(model, model_filepath):
    '''
    model [pipeline] = adaboost classifier model
    model_filepath [string] = path of where to save the model file
    Function to save the model as a pickle file.
    '''
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    '''
    Function to load data, build, train, evaluate and save the model.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        cv = build_model()
        
        print('Training model...')
        cv.fit(X_train, Y_train)
        model = cv.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'models/train_classifier.py data/DisasterResponse.db models/classifier.pkl')


if __name__ == '__main__':
    main()