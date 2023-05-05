import json
import random
import re
import sys

import joblib
import keras_preprocessing.text
import numpy as np
import pandas as pd
import tensorflow
from flask import Flask, jsonify, request
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

tensorflow.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)

app = Flask(__name__)

model = keras.models.load_model("./models/saved/lstm.h5")
with open('./models/saved/tokenizer.json') as f:
    data = json.load(f)
    tokenizer = keras_preprocessing.text.tokenizer_from_json(data)
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("./models/saved/labels.npy", allow_pickle=True)


@app.route("/", methods=["POST"])
def index():
    print("someone sent a post to /")
    input_json = request.get_json(force=True)
    print(input_json)
    text = [clean_text(input_json["text"])]
    print(text)
    text = tokenizer.texts_to_sequences(text)
    print(text)
    text = keras.preprocessing.sequence.pad_sequences(text, maxlen=100)
    print(text)
    pred_label = model.predict(text)
    pred_label_dec = label_encoder.inverse_transform([pred_label.argmax(axis=-1)])
    pred_label_prob = pred_label.max(axis=-1)
    response = {"label": pred_label_dec.tolist(),
                "prob": pred_label_prob.tolist()}
    return jsonify(response)


@app.route("/test", methods=["POST"])
def hello_world():
    print("someone sent a post to /test")
    return jsonify({"hello": "world"})


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
