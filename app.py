import json
import random
import re
import sys
import time

import flask
import joblib
import keras_preprocessing.text
import numpy as np
import pandas as pd
import tensorflow
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

tensorflow.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/predict", methods=["POST"])
@cross_origin()
def index():
    start_time = time.time()
    print(f"Start: {time.time() - start_time}")
    model = keras.models.load_model("./models/saved/lstm.h5")
    print(f"Loading model: {time.time() - start_time}")
    with open('./models/saved/tokenizer.json') as f:
        data = json.load(f)
        tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
    print(f"Loading tokenizer: {time.time() - start_time}")
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)
    print(f"Loading label encoder: {time.time() - start_time}")

    input_json = request.get_json(force=True)
    text = [clean_text(input_json["text"])]
    text = tokenizer.texts_to_sequences(text)
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=100)
    print(f"Preprocessing: {time.time() - start_time}")

    pred_label = model.predict(text)
    print(f"Predicting: {time.time() - start_time}")

    pred_label_dec = label_encoder.inverse_transform([pred_label.argmax(axis=-1)])
    pred_label_prob = pred_label.max(axis=-1)
    response_body = {"label": pred_label_dec.tolist(),
                     "prob": pred_label_prob.tolist()}
    print(f"Response: {time.time() - start_time}")
    # response = flask.Response(response=json.dumps(response_body), status=200)
    # response.headers["Access-Control-Allow-Origin"] = "http://localhost:3000"
    # response.headers["Access-Control-Allow-Credentials"] = "true"
    # response.headers["Content-Type"] = "application/json"
    # response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    # response.headers["Access-Control-Allow-Headers"] = "Origin, Content-Type, Accept"
    # print(response.headers)
    # print(response.json)
    response = flask.jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route("/test", methods=["GET"])
def hello_world():
    response = flask.jsonify({'hello': 'world'})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(r'^(\d{1,2})(.|\)) ', '', text)
    text = re.sub(r'  ', ' ', text)
    return text
