import io
import json
from abc import ABC
import random

import numpy as np
import scipy
import tensorflow
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TextVectorization, Bidirectional
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.saving import load_model
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from keras.layers import Embedding, GlobalMaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from data.GloVeEmbedding import GloVeEmbedding
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

tensorflow.keras.utils.set_random_seed(42)
random.seed(42)
np.random.seed(42)

class LSTM:
    def __init__(self, dataset, load_model_path=None, save_tokenizer=None):
        # super().__init__(dataset)
        self.__dataset = dataset
        self.__train = dataset.train
        self.__test = dataset.test
        self.__vocab_size = dataset.vocab_size
        self.__tokenizer = Tokenizer()
        self.__tokenizer.fit_on_texts(self.__train["description"])

        if load_model_path is not None:
            self.__model = load_model(load_model_path)
        if save_tokenizer is not None:
            tokenizer_json = self.__tokenizer.to_json()
            with io.open(save_tokenizer, 'w+', encoding='utf-8') as f:
                f.write(json.dumps(tokenizer_json, ensure_ascii=False))

    def train(self, embedding):
        train_seq = self.__tokenizer.texts_to_sequences(self.__train["description"])
        train_seq = pad_sequences(train_seq, maxlen=100)
        vocab_size = len(self.__tokenizer.word_index) + 1
        embedding_matrix = embedding.compute_embedding_matrix(self.__tokenizer, vocab_size)

        model = Sequential()
        model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False))
        model.add(LSTMLayer(128, dropout=0.5, recurrent_regularizer='l2', stateful=False))
        # model.add(GlobalMaxPooling1D())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(16, activation='softmax'))

        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["acc"])
        print(model.summary())

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        model.fit(np.array(train_seq), np.array(self.__train["label"]), batch_size=64,
                  validation_split=0.2, epochs=1000, verbose=1, callbacks=[es])
        model.save("./models/saved/lstm.h5", save_format="h5")
        self.__model = model

    def predict(self, text):
        text = self.__dataset.clean_text(text)
        text = self.__tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=100)
        pred_probs = self.__model.predict(text)
        pred_max_prob = pred_probs.max(axis=-1)
        pred_label = self.__dataset.decode_label(pred_probs.argmax(axis=-1))
        return pred_label, pred_max_prob, pred_probs

    def evaluate(self):
        test_seq = self.__tokenizer.texts_to_sequences(self.__test["description"])
        test_seq = pad_sequences(test_seq, maxlen=100)
        score = self.__model.evaluate(test_seq, self.__test["label"], verbose=0)
        print(f"Test loss: {score[0]}")
        print(f"Test accuracy: {score[1]}")

    def plot_confusion_matrix(self):
        test_seq = self.__tokenizer.texts_to_sequences(self.__test["description"])
        test_seq = pad_sequences(test_seq, maxlen=100)
        pred_labels = self.__model.predict(test_seq)
        pred_labels = [y.argmax(axis=-1) for y in pred_labels]
        matrix = confusion_matrix(self.__test["label"], pred_labels)
        labels = self.__dataset.decode_label(range(self.__test["label"].nunique()))
        cm_display = ConfusionMatrixDisplay(matrix, display_labels=labels)
        cm_display.plot(xticks_rotation=60)
        plt.show()
