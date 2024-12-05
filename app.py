import re
import string
import sqlite3 as sql
import tensorflow as tf
import pandas as pd
from flasgger import LazyJSONEncoder, LazyString, Swagger, swag_from
from flask import Flask, jsonify, request
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from cleantext import clean
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import numpy as np
import pickle
from keras.models import load_model
from joblib import dump, load
from tensorflow.keras.preprocessing.sequence import pad_sequences

factory_stopwords = StopWordRemoverFactory()
stopword = factory_stopwords.create_stop_word_remover()
factory_stems = StemmerFactory()
stemmers = factory_stems.create_stemmer()
set_stopwords_nltk = set(stopwords.words('indonesian'))
stopwords_nltk = list(set_stopwords_nltk)
custom_word = ['nya', 'sih', 'dan', 'yang', 'ada', 'di', 'ini', 'dari', 'jadi', 'ke', 'itu', 'sana', 'sini', 'ya', 'nih', 'iya']
all_stopwords = stopwords_nltk + custom_word
dictionary = ArrayDictionary(all_stopwords)
stopwordremover = StopWordRemover(dictionary)

#Declarating the global variables for database 
connection = sql.connect('main.db', check_same_thread=False)        #Make connection to database
cursor = connection.cursor()
alay  = pd.read_sql('select * from Kamus_alay', connection)         #Read kamus alay from database 
dict_alay = dict(alay.values)       


app = Flask(__name__)
app.json_encoder = LazyJSONEncoder
swagger_template = dict(
    info = {
        'title': LazyString(lambda: 'API for Machine Learning using Neural Networks and LSTM Method'),
        'version': LazyString(lambda: '1.0.0'),
        'description': LazyString(lambda: 'API Documentation for Machine Learning of Sentiment Analysis taken from both text and file'),
    },
    host = LazyString(lambda: request.host)
)
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger = Swagger(app, template=swagger_template,config=swagger_config)


@swag_from("docs/home.yml", methods=['GET'])
@app.route('/', methods=['GET'])
#Function for showing home page
def home():
    json_response = {
        'status_code': 200,
        'description': "API Text and File for Predicting Sentiment using Neural Network and LSTM",
        }
    response_data = jsonify(json_response)
    return response_data

@swag_from("docs/all_data.yml", methods=['GET'])
@app.route('/all_data', methods=['GET'])
#Function for showing all data from database
def database():
    all_data = pd.read_sql('SELECT * from Predicting_sentiment',connection)
    all_data = all_data.T.to_dict()
    return jsonify(
        all_data = all_data,
        status_code=200
    )

@swag_from("docs/text_lstm.yml", methods=['POST'])
@app.route('/text_lstm', methods=['POST'])
#Function for text cleaning from input text
def text_lstm():
    text = request.form['raw_text']
    """This function is used to clean the text"""
    def cleaningText(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
        text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
        text = re.sub(r'RT[\s]', '', text) # remove RT
        text = re.sub(r"http\S+", '', text) # remove link
        text = re.sub(r'[0-9]+', '', text) # remove numbers
        text = text.replace('\n', ' ') # replace new line into space
        text = ' '.join(dict.fromkeys(text.split())) #Remove repetitive
        text = ' '.join(dict_alay.get(alay_word,alay_word) for alay_word in text.split())
        text = text.translate(str.maketrans('','', string.punctuation))
        return text
    
    """This function is used to lower the text"""
    def casefoldingText(text):
        text = text.lower() 
        return text
    """This function is used to tokenize the text"""
    def tokenizingText(text):
        text = word_tokenize(text) 
        return text
    
    """This function is used to remove stopwords the text"""
    def removestopWord(text):
        text = ' '.join([i for i in text if i not in all_stopwords])
        return text
    
    """This function is used to stem the the text"""
    def stemmingText(text):
        text = stemmers.stem(text)
        return text

    """Appy text to the function"""
    clean_text = cleaningText(text)
    clean_text = casefoldingText(clean_text)
    clean_text = tokenizingText(clean_text)
    clean_text = removestopWord(clean_text)
    clean_text = stemmingText(clean_text)

    """This function is used to perdict the text by loading the tokenizer.pickle 
        load the model and predict the input text"""
    def predictingSentiment(clean_text):
        with open('LSTM_Model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        model = load_model('LSTM_Model/best_model_LSTM.h5')

        max_len = 43
        clean_text = tokenizer.texts_to_sequences([clean_text])
        pad = pad_sequences(clean_text, maxlen=max_len, padding='post')

        prediction = model.predict(pad)
        labels = ['negative', 'neutral', 'positive']
        prediction = prediction.argmax()
        return labels[prediction]

    """Call the function"""
    sentiment_label = predictingSentiment(clean_text)

    """Put the data to database"""
    query = f"INSERT INTO Predicting_sentiment (Raw_text, Clean_text, Sentiment) VALUES ('{text}','{clean_text}', '{sentiment_label}')"
    cursor.execute(query)
    connection.commit()

    #Return cleaned_data to jsonify
    return jsonify(
        raw_text = text,
        clean_text = clean_text,
        sentiment = sentiment_label,
        status_code=200
    )


@swag_from("docs/file_lstm.yml", methods=['POST'])
@app.route('/file_lstm', methods=['POST'])
#Function for file cleaning from file
def file_lstm():
    file = request.files['data_file']
    file = pd.read_csv(file, encoding='latin-1')
    file = file.head()
    """This function is used to clean the text"""
    def cleaningText(text):
        text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
        text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
        text = re.sub(r'\\x[A-Za-z0-9./]+', '', text)
        text = re.sub(r'RT[\s]', '', text) # remove RT
        text = re.sub(r"http\S+", '', text) # remove link
        text = re.sub(r'[0-9]+', '', text) # remove numbers
        text = text.replace('\n', ' ') # replace new line into space
        text = ' '.join(dict.fromkeys(text.split())) #Remove repetitive
        text = ' '.join(dict_alay.get(alay_word,alay_word) for alay_word in text.split())
        text = text.translate(str.maketrans('','', string.punctuation))
        return text
    """This function is used to lower the text"""
    def casefoldingText(text):
        text = text.lower() 
        return text
    """This function is used to tokenize the text"""
    def tokenizingText(text):
        text = word_tokenize(text) 
        return text
    
    """This function is used to remove stopwords the text"""
    def removestopWord(text):
        text = ' '.join([i for i in text if i not in all_stopwords])
        return text
    """This function is used to stem the text"""
    def stemmingText(text):
        text = stemmers.stem(text)
        return text
    
    """Appy file to the function"""
    file['Tweet_Clean'] = file.Tweet.apply(cleaningText)
    file['Tweet_Clean'] = file.Tweet_Clean.apply(casefoldingText)
    file['Tweet_Clean_Preprocessed'] = file.Tweet_Clean.apply(tokenizingText)
    file['Tweet_Clean_Preprocessed'] = file.Tweet_Clean_Preprocessed.apply(removestopWord)
    file['Tweet_Clean_Preprocessed'] = file.Tweet_Clean_Preprocessed.apply(stemmingText)
    file.drop_duplicates(subset = 'Tweet_Clean', inplace = True)
    file.dropna(subset = ['Tweet_Clean'], inplace = True)

    """This function is used to perdict the text by loading the tokenizer.pickle 
        load the model and predict the input text"""
    def predictingSentiment(file_clean):
        with open('LSTM_Model/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        model = load_model('LSTM_Model/best_model_LSTM.h5')

        max_len = 43
        clean_text = tokenizer.texts_to_sequences([file_clean])
        pad = pad_sequences(clean_text, maxlen=max_len, padding='post')

        prediction = model.predict(pad)
        labels = ['negative', 'neutral', 'positive']
        prediction = prediction.argmax()
        return labels[prediction]

    """Call the function and apply to all text on the file, the put on list called sentiment"""
    sentiment = []
    for text in range(len(file['Tweet_Clean_Preprocessed'])):
        sentiment_label = predictingSentiment(file['Tweet_Clean_Preprocessed'][text])
        sentiment.append(sentiment_label)
    
    """Make all lists become dataframe"""
    raw_text = file['Tweet'].tolist()
    clean_text = file['Tweet_Clean_Preprocessed'].tolist()
    sentiment = sentiment
    combined_data = list(zip(raw_text, clean_text, sentiment))
    total = [{'Raw_text': text[0], 'Clean_text': text[1], 'Sentiment': text[2]} for text in combined_data]
    predicting_sentiment = pd.DataFrame(total)
    predicting_sentiment = predicting_sentiment.T.to_dict() 

    return jsonify(
        predicting_sentiment=predicting_sentiment,
        status_code=200
    )


if __name__ == '__main__':
    app.run(debug=True)
