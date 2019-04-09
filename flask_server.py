from flask import Flask, render_template, jsonify, request, session, redirect, url_for, flash
from flask_wtf import FlaskForm
from flask_cors import CORS
from flask_bootstrap import Bootstrap
from wtforms import StringField, SubmitField
from wtforms.validators import Required

from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras import backend as K
import tensorflow as tf

app = Flask(__name__,template_folder='flask_templates')
Bootstrap(app)
CORS(app)

from sklearn.model_selection import train_test_split
import pandas as pd

from io import StringIO
import nltk
import string
import os
import glob
import json
import datetime
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import pickle
import shutil

tokenizer = None
lstm_model = None
max_length = 100
graph = None
model_version = None

app.config['SECRET_KEY'] = 'hard to guess string'

def clean_text(text):
    if text is None:
        return ""
    elif isinstance(text,float):
        if str(text) == 'nan':
            #print('nan')
            return ""

    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    #print(text)
    return text

def get_lstm_model_function(vocabulary_size):

    def create_lstm_model(optimizer='adam'):
        model_lstm = Sequential()
        model_lstm.add(Embedding(vocabulary_size, 100, input_length=max_length))
        model_lstm.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model_lstm.add(Dense(1, activation='sigmoid'))
        model_lstm.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model_lstm
    
    return create_lstm_model

def formatDataframe(df):
    df.fillna('')
    print(df[:100])
    df['combined_text'] = df.apply(lambda x: x.OtherSx2, axis=1)
    df['combined_text_clean'] = df['combined_text'].map(lambda x : clean_text(x))
    return df

@app.route('/trainingCSVHeaders', methods=['GET'])
def getTrainingCSVHeaders():
    responses = jsonify({"headers":"examDescription,OtherHx,OtherSx,OtherSx2,indicationDescription,label_binary"})
    responses.status_code = 200
    return responses

@app.route('/load_model', methods=['POST'])
def loadModel():
    global lstm_model, tokenizer, graph, model_version
    modeldata_json = request.get_json()
    files = glob.glob('data/*_data')

    if modeldata_json['model_version'] in map(lambda x: x.replace('data/',''),files):
        try:
            print("Loading Pickle : data/" +  modeldata_json['model_version'] + "/" + modeldata_json['model_version'] + "_tokenizer_combined.pickle")
            with open('data/' + modeldata_json['model_version'] + '/' + modeldata_json['model_version'] + '_tokenizer_combined.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
                print(tokenizer)
        except Exception as e:
            print('Could not open tokenizer')
            print(e)
            responses = jsonify({"message":"undo error (tokenizer), please contact admin."})
            responses.status_code = 500
            return(responses)
            raise e
        K.clear_session()
        graph = tf.get_default_graph()
        try:
            print("Loading LSTM Model : data/" + modeldata_json['model_version'] + "/" + modeldata_json['model_version'] + "_combined.h5")
            lstm_model = load_model('data/' + modeldata_json['model_version'] + '/' + modeldata_json['model_version'] + '_combined.h5')
            #update model_version
            model_version = "ml_postproc_lstm_" + modeldata_json['model_version']
        except Exception as e:
            print('Could not open lstm_model')
            print(e)
            responses = jsonify({"message":"undo error (lstm_model), please contact admin."})
            responses.status_code = 500
            return(responses)
            raise e

        responses = jsonify({"message":"loading of model: " + model_version + " complete."})
        responses.status_code = 200
        return(responses)
        raise e

    else:
        responses =  jsonify({"message": modeldata_json['model_version'] + " not found."})
        responses.status_code = 500
        return responses

@app.route('/model_version', methods=['GET'])
def getModelVersion():
    global model_version
    responses =  jsonify({"message":model_version})
    responses.status_code = 200
    return responses

@app.route('/undo_retrain', methods=['POST'])
def undoRetrain():
    global tokenizer, graph, lstm_model, model_version
    K.clear_session()

    files = glob.glob('data/*_data')
    files.sort(reverse=True)
    responses = None

    if len(files) == 0:
        responses = jsonify({"messsage":"no models available to undo. model_version remains: " + model_version})
        responses.status_code = 500
        return(responses)

    if len(files) == 1:
        shutil.rmtree(files[0])
        responses = jsonify({"messsage":"undo complete. model_version remains: " + model_version})
        responses.status_code = 200
        return(responses)

    try:
        print("Loading Pickle : " +  files[1] + "/" + files[1].replace('data/','') + "_tokenizer_combined.pickle")
        with open(files[1] + '/' + files[1].replace('data/','') + '_tokenizer_combined.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            print(tokenizer)
    except Exception as e:
        print('Could not open tokenizer')
        print(e)
        responses = jsonify({"message":"undo error (tokenizer), please contact admin."})
        responses.status_code = 500
        return(responses)
        raise e
    graph = tf.get_default_graph()
    try:
        print("Loading LSTM Model : " + files[1] + "/" + files[1].replace('data/','') + "_combined.h5")
        lstm_model = load_model(files[1] + '/' + files[1].replace('data/','') + '_combined.h5')
        #update model_version
        model_version = "ml_postproc_lstm_" + files[1].replace('data/','')
    except Exception as e:
        print('Could not open lstm_model')
        print(e)
        responses = jsonify({"message":"undo error (lstm_model), please contact admin."})
        responses.status_code = 500
        return(responses)
        raise e
    
    try:
        shutil.rmtree(files[0])
    except e:
        print('Could not delete previous model')
        responses = jsonify({"message":"undo error (file removal), please contact admin."})
        responses.status_code = 500
        return(responses)
        raise e

    
    responses = jsonify({"message":"undo complete. model_version: " + model_version})
    responses.status_code = 200
    return(responses)

@app.route('/retrain', methods=['POST'])
def retrain():
    global graph, tokenizer, lstm_model, model_version
    K.clear_session()
    #tf.reset_default_graph()
    #graph = tf.get_default_graph()
    try:
        csv_data = None
        json_data = None

        if request.is_json:
            json_data = json.loads(request.get_json())['data']
        else:
            csv_data = request.files['data']

        now_date_folder_name = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_data"

        new_data_df = None
        previous_data_df = None

        if (not os.path.isdir('data/' + now_date_folder_name)):

            #if 'model' in model_csv_json:
            os.makedirs('data/' + now_date_folder_name)
            #print(model_csv_json['model'])
            #new_data_df = pd.read_json(model_csv_json['data'])
            if csv_data:
                #print(csv_data)
                new_data_df = pd.read_csv(csv_data)
            else:
                new_data_df = pd.read_json(json_data)

            new_data_df.to_csv('data/' + now_date_folder_name + "/" + now_date_folder_name + ".csv",index=False)
            #else:
                #return getTrainingHeaders()


            files = glob.glob('data/*_data')
            files.sort(reverse=True)
            print(files)

            #files[0].replace('data/','')
            if len(files) > 1 and files[0] == 'data/' + now_date_folder_name:
                print(files[1])
                previous_data_df = pd.read_csv(files[1] + "/" + files[1].replace('data/','') + "_combined.csv")
                new_data_df = pd.concat([new_data_df,previous_data_df])
            
            new_data_df.to_csv('data/' + now_date_folder_name + "/" + now_date_folder_name + "_combined.csv",index=False)

            new_data_df = formatDataframe(new_data_df)

            max_length = 100
            new_tokenizer = Tokenizer()
            new_tokenizer.fit_on_texts(new_data_df['combined_text_clean'])
            vocabulary_size = len(new_tokenizer.word_index) + 1
            sequences = new_tokenizer.texts_to_sequences(new_data_df['combined_text_clean'])
            lstm_data = pad_sequences(sequences, maxlen=max_length)

            new_lstm_model = KerasClassifier(build_fn=get_lstm_model_function(vocabulary_size), epochs=3, verbose=0)
            X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(lstm_data, new_data_df.label_binary, new_data_df.index, test_size=0.10, random_state=0)
            new_lstm_model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=3)

            new_lstm_model.model.save('data/' + now_date_folder_name + '/' + now_date_folder_name + '_combined.h5')
            with open('data/' + now_date_folder_name + '/' + now_date_folder_name + '_tokenizer_combined.pickle','wb') as handle: 
                pickle.dump(new_tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL) 

            print("Re-Loading LSTM Model : data/" + now_date_folder_name + '/' + now_date_folder_name + "_combined.h5")
            lstm_model = load_model('data/' + now_date_folder_name + '/' + now_date_folder_name + '_combined.h5')
            model_version = "ml_postproc_lstm_" + now_date_folder_name
            #lstm_model = new_lstm_model
            tokenizer = new_tokenizer
            graph = tf.get_default_graph()

    except Exception as e:
        raise e
    responses = jsonify({"message":"retraining complete. model_version: " + model_version})
    responses.status_code = 200
    return(responses)

def getPrediction(examDescription="", OtherHX="", OtherSX="", OtherSX2="", indicationDescription=""):
    global lstm_model, tokenizer, max_length, graph, model_version
    #to_classify = examDescription + '. ' + OtherSX + '. ' + indicationDescription
    #to_classify = examDescription + '. ' + OtherHX + '. ' + OtherSX + '. ' + indicationDescription
    to_classify = OtherSX2
    to_classify = clean_text(to_classify)
    to_classify = pad_sequences(tokenizer.texts_to_sequences([to_classify]),maxlen=max_length)
    print(to_classify)

    prediction = [[]]
    with graph.as_default():
        prediction  = lstm_model.predict(to_classify)
        #prediction  = lstm_model.predict_proba(to_classify)
        #prediction  = 1.0 - float(prediction[0][0])
        prediction  = prediction[0][0]
        print(prediction)
    #return(str(1.0-prediction[0][0]))
    return(str(prediction))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        orderdata_json = request.get_json()
        print(orderdata_json)
    except Exception as e:
        raise e
    prediction = getPrediction(examDescription=orderdata_json['examDescription'],OtherSX=orderdata_json['OtherSX'],OtherHX=orderdata_json['OtherHX'],OtherSX2=orderdata_json['OtherSX2'],indicationDescription=orderdata_json['indicationDescription'])
    #prediction = getPrediction(examDescription=orderdata_json['examDescription'],OtherSX=orderdata_json['OtherSX'],OtherSX2=orderdata_json,OtherHX=orderdata_json['OtherHX'],indicationDescription=orderdata_json['indicationDescription'])
    responses = jsonify({"prediction":prediction, "model_version": model_version})
    responses.status_code = 200
    return(responses)

class PredictionForm(FlaskForm):
    examDescription = StringField('examDescription', validators=[Required()])
    OtherHX = StringField('OtherHX', validators=[Required()])
    OtherSX = StringField('OtherSX', validators=[Required()])
    indicationDescription = StringField('indicationDescription', validators=[Required()])
    #prediction = StringField('prediction', validators=[Required()])
    submit = SubmitField('Submit')

@app.route('/', methods=['GET', 'POST'])
def index():
    name = None
    form = PredictionForm()
    if form.validate_on_submit():
        old_examDescription = session.get('examDescription')
        old_OtherHX = session.get('OtherHX')
        old_OtherSX = session.get('OtherSX')
        old_indicationDescription = session.get('indicationDescription')
        if (old_examDescription is not None and old_examDescription != form.examDescription.data) or (old_OtherSX is not None and old_OtherSX != form.OtherSX.data) or (old_indicationDescription is not None and old_indicationDescription != form.indicationDescription) or (old_OtherHX is not None and old_OtherHX != form.OtherHX.data) :
            flash('data has changed, predicting')

        session['examDescription'] = form.examDescription.data
        session['OtherHX'] = form.OtherHX.data
        session['OtherSX'] = form.OtherSX.data
        session['indicationDescription'] = form.indicationDescription.data
        session['prediction'] = getPrediction(examDescription=form.examDescription.data,OtherHX=form.OtherHX.data,OtherSX=form.OtherSX.data,indicationDescription=form.indicationDescription.data)
        #form.examDescription.data = ''
        #form.OtherSX.data = ''
        #form.indicationDescription.data = ''
        return redirect(url_for('index'))
    #return render_template('index.html', form=form, examDescription=session.get('examDescription'), OtherSX=session.get('OtherSX'), indicationDescription=session.get('indicationDescription'),prediction=session.get('prediction'))
    return render_template('index.html', form=form, examDescription=session.get('examDescription'),OtherHX=session.get('OtherHX'),OtherSX=session.get('OtherSX'), indicationDescription=session.get('indicationDescription'),prediction=session.get('prediction'))

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server...please wait until server has fully started"))
    files = glob.glob('data/*_data')
    files.sort(reverse=True)
    try:
        if len(files) > 0:
            print("Loading Pickle : " +  files[0] + "/" + files[0].replace('data/','') + "_tokenizer_combined.pickle")
            with open(files[0] + '/' + files[0].replace('data/','') + '_tokenizer_combined.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
                print(tokenizer)
    except Exception as e:
        print('Could not open tokenizer')
        raise e
    graph = tf.get_default_graph()
    try:
        if len(files) > 0:
            print("Loading LSTM Model : " + files[0] + "/" + files[0].replace('data/','') + "_combined.h5")
            lstm_model = load_model(files[0] + '/' + files[0].replace('data/','') + '_combined.h5')
            model_version = "ml_postproc_lstm_" + files[0].replace('data/','')
    except Exception as e:
        print('Could not open lstm_model')
        raise e
    """
    try:
        with open('../tokenizer_examDescription_OtherSX_OtherHX_indicationDescription.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except Exception as e:
        print('Could not open tokenizer')
        raise e
    graph = tf.get_default_graph()
    try:
        #lstm_model = load_model('../lstm_only_examDescription_OtherSX_indicationDescription.h5')
        lstm_model = load_model('../lstm_examDescription_OtherSX_OtherHX_indicationDescription.h5')
    except Exception as e:
        print('Could not open lstm_model')
        raise e
    """
    app.run(host="0.0.0.0",port=8484,threaded=True)
