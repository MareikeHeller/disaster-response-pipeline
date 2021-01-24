import sys
sys.path.append('../')
import json
import plotly
import pandas as pd
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
from modules.utils import tokenize, SentenceCountExtractor

app = Flask(__name__)

# load data
try:
    # heroku
    engine = create_engine('sqlite:///./data/DisasterResponse.db')
    df = pd.read_sql_table('labeled_messages', engine)
except:
    # local
    engine = create_engine('sqlite:///../data/DisasterResponse.db')
    df = pd.read_sql_table('labeled_messages', engine)

# load model
global model
try:
    # heroku
    model = joblib.load("./models/classifier.pkl")
except:
    # local
    model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    ### GRAPHS ###
    # 1. Labeled Messages per Category
    graph_one = []
    # extract data needed for visuals
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_means = df.iloc[:, 4:].mean().sort_values(ascending=False).apply(lambda x: '{:.0f}%'.format(x * 100))
    category_names = list(category_counts.index)

    # create visuals
    graph_one.append(
        Bar(
            x=category_names,
            y=category_counts,
            text=category_means,
            textposition='outside',
        )
    )

    layout_one = dict(title='Labeled Messages per Category',
                      xaxis=dict(title='Category', tickangle=25),
                      yaxis=dict(title='Count'),
                      )

    # 2. Labeled Messages per Category by Genre
    graph_two = []
    # extract data needed for visuals
    # group by genre and count messages per category
    category_counts_grouped = df.groupby('genre').sum()[category_names].reset_index()
    # pivot all categories from columns to rows
    category_counts_by_genre = pd.melt(category_counts_grouped, id_vars='genre', value_vars=category_names)
    genrelist = category_counts_by_genre['genre'].unique().tolist()

    # create visuals
    for genre in genrelist:
        graph_two.append(
            Scatter(
                x=category_counts_by_genre[category_counts_by_genre['genre'] == genre]['variable'].tolist(),
                y=category_counts_by_genre[category_counts_by_genre['genre'] == genre]['value'].tolist(),
                mode='lines',
                name=genre
            )
        )

    layout_two = dict(title='Labeled Messages per Category by Genre',
                      xaxis=dict(title='Category', tickangle=25),
                      yaxis=dict(title='Count'),
                      )

    # 3. Distribution of Message Genres
    graph_three = []
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # create visuals
    graph_three.append(
        Bar(
            x=genre_names,
            y=genre_counts
        )
    )

    layout_three = dict(title='Distribution of Message Genres',
                        xaxis=dict(title='Genre'),
                        yaxis=dict(title='Count')
                        )

    # Append all charts to the figures list
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    graphs.append(dict(data=graph_three, layout=layout_three))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
