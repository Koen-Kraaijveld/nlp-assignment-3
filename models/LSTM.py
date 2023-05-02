from abc import ABC

import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TextVectorization
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras import Input
from keras.layers import Embedding

from .Model import Model


# print(tf.config.list_physical_devices('GPU'))

class LSTM(Model, ABC):
    def __init__(self, dataset, use_tfidf=False):
        super().__init__(dataset)
        self.__train = dataset.train
        self.__vocab_size = dataset.vocab_size
        self.__use_tfidf = use_tfidf
        self.__model = self.__design_model()

    def __process_to_tfidf(self):
        vectorize = TextVectorization(max_tokens=self.__vocab_size, output_mode="tf-idf")
        print(self.__train["description"].shape)
        vectorize.adapt(self.__train["description"])
        return vectorize

    def __design_model(self):
        model = Sequential()
        if self.__use_tfidf:
            vectorize = self.__process_to_tfidf()
            self.__train = tensorflow.data.Dataset.from_tensor_slices((self.__train["description"],
                                                                       self.__train["label"]))
            self.__train = self.__train.batch(64).map(lambda x, y: (vectorize(x), y))
            model.add(Input(shape=(vectorize.vocabulary_size(),)))
            model.add(vectorize)
        model.add(Embedding(self.__vocab_size, 100, input_length=100))
        model.add(LSTMLayer(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(345, activation='softmax'))
        return model

    def train(self):
        self.__model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.__model.fit(self.__train.batch(64), epochs=20)

    def evaluate(self, dataset):
        pass


# self.dataset = dataset
        # self.tokenizer = Tokenizer(num_words=self.dataset.vocab_size)
        # self.tokenizer.fit_on_texts(self.dataset.train["description"])
        # sequences = self.tokenizer.texts_to_sequences(self.dataset.train["description"])
        # self.data = pad_sequences(sequences, maxlen=100)