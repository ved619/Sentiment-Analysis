from dash_html_components.H2 import H2
import pandas as pd
import dash
import dash_html_components as html
import webbrowser
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text  import TfidfTransformer
from nltk.stem import WordNetLemmatizer
import pickle
import nltk
import re

stop_words=nltk.corpus.stopwords.words('english')
lemm_text=WordNetLemmatizer()

app=dash.Dash(external_stylesheets=[dbc.themes.CYBORG])
project_name=None

def app_ui():
    main_layout=html.Div(
    [
        html.H1(children="Sentiment Analysis",style={'text-align':'center'}),
        dbc.Input(
            id="text-area",
            placeholder="Enter text",
            type='text',
            style={'width':'30%','height':'8%','margin':'auto'}
        ),
        html.Br(),
        dbc.Alert(id='v-area',children="Prediction",color='danger',
        style={'width':'30%','height':'40%','margin':'0 auto','top':'160px'}
        ),
    ]

    )
    return main_layout

@app.callback(
    [
        Output("v-area","color"),
        Output('v-area','children'),
    ],
    [
        Input("text-area","value"),
    ]
)
def update_appui(text_value):
    print()
    text_value=check_review(text_value)
    if text_value=='Positive':
        return "success",text_value
    else:
        return "danger",text_value

def open_browser():
    webbrowser.open_new_tab("http://127.0.0.1:8050/")


def preprocess_text(text):
    text=text.lower()
    x=re.findall(r"[a-zA-Z]+",text)
    x=[i for i in x if i not in stop_words]
    x=" ".join([lemm_text.lemmatize(i) for i in x])
    return x


def check_review(text):
    text=preprocess_text(text)
    f=open("model.pkl",'rb')
    r_model=pickle.load(f)
    v=open('vect.pkl','rb')
    vect=pickle.load(v)
    s=int(r_model.predict(vect.transform([text])))
    if s==1:
        return "Positive"
    else:
        return "Negative"


def main():
    print("start of your project")
    global app
    global project_name
    project_name="Sentiment analysis"

    app.title=project_name
    app.layout=app_ui()
    open_browser()
    app.run_server()
    
    print("End of the project")
    project_name=None
    app=None


if __name__=='__main__':
    main()