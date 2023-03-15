# Disaster Response Pipeline

Heroku free plan was discontinued. Web app to be migrated to another hosting platform.

_Find the web app: https://disaster-response-pipeline-mh.herokuapp.com/_

## Table of Contents
1. [Instructions](#instructions)
2. [Installation](#installation)
3. [Objective](#objective)
4. [File Descriptions](#file-descriptions)
5. [Licensing, Authors, Acknowledgements](#licensing-authors-acknowledgements)

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

	- To run ETL pipeline that cleans data and stores in database `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	- To run ML pipeline that trains classifier and saves `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
	- Run the following command in the app's directory to run your web app. `python run.py`

2. Go to http://0.0.0.0:3001/

## Installation

The code was developed using Python 3.6.3. Necessary packages beyond the Python Standard Library are:
- Flask==0.12.5
- gunicorn==19.10.0
- joblib==1.0.0
- nltk==3.2.5
- numpy==1.19.5
- pandas==0.23.3
- plotly==2.0.15
- scikit-learn==0.19.1
- SQLAlchemy==1.2.19

The environment can be installed using [requirements.txt](https://github.com/MareikeHeller/disaster-response-pipeline/blob/main/requirements.txt).

### Deployment
The application is created using Flask and was deployed on Heroku. 

Heroku free plan was discontinued. Web app to be migrated to another hosting platform.

_You can access the app here: https://disaster-response-pipeline-mh.herokuapp.com/_ 

## Objective
This web app provides an interface for the automatic classification of disaster response messages into category labels.
It includes a natural language processing pipeline using a
- count vectorizer
- TF-IDF transformer
- customized transformer of sentence counts (SentenceCountExtractor)
- Random Forest Classifier

### Details
A new message can be assigned to label categories based on the underlying classifier model by inserting it the input line.
![](https://github.com/MareikeHeller/disaster-response-pipeline/blob/main/screenshots/classify_message.PNG)
![](https://github.com/MareikeHeller/disaster-response-pipeline/blob/main/screenshots/classification_result.PNG)

The main page shows three visualizations (plotly) on the training dataset, including:
- Labeled Messages per Category
	- *categories are medical_products, water, security, buildings, missing_people and more*
- Labeled Messages per Category by Genre
	- *genres are direct, news & social*
- Distribution of Message Genres

![](https://github.com/MareikeHeller/disaster-response-pipeline/blob/main/screenshots/visualization_example.PNG)

In case new data becomes available for model training in the future in order to improve the classification performance, the ETL & ML pipelines can be executed as described under [Instructions](#instructions). 

## File Descriptions
**app**
- templates
	- master.html & go.html
    	- *app HTML functions & design*
- run.py
	- *runs the app*
    - *loads data from database and model from pickel file (different directories for local vs. web deployment are handled using try except statements)*
    - *note: from sklearn.externals import joblib was replaced by direct import joblib*
    - *creates plotly visualizations*
    
**data**
- disaster_categories.csv
	- *raw data on categories (used as labels)*
- disaster_messages.csv
	- *raw data on messages (after transformation used as features)*
- process_data.py
	- *loads and cleans raw data*
    - *note: label counts > 1 were set to 1*
    - *saves the resulting clean data in a database with one table*
- DisasterResponse.db
	- *final clean database with one table "labeled_messages"*
    
**models**
- train_classifier.py
	- *loads data and builds the model using an ML pipeline*
    - *including GridSearch cross-validation and evaluation of the model*
    - *saves the resulting model as pickle file*
![](https://github.com/MareikeHeller/disaster-response-pipeline/blob/main/screenshots/model_metrics.PNG)
- classifier.pkl
	- *final classifier model*
    
**modules**
- utils.py
	- *contains function "tokenize" and class "SentenceCountExtractor"*
    - *tokenize: Transforms text messages (documents) into normalized & lemmatized tokens*
    - *SentenceCountExtractor: Customized transformer based on whether a message contains one or multiple sentences. Used as a boolean feature in the ML model.*
    - *was necessary to isolate from train_classifier.py & run.py for deployment with heroku*
    
**Procfile**
- *\[heroku\] app deployment*

**nltk.txt**
- *\[heroku\] contains a list of nltk corpora to be downloaded for model to classify messages*

**requirements.txt**
-  *can be used to install the python environment*

## Licensing, Authors, Acknowledgements
This web app was developed during an exercise related to the [Udacity Data Science Nanodegree](https://www.udacity.com/school-of-data-science).

The data used in this project was kindly provided by [Figure Eight (now Appen)](https://appen.com/).
