from abc import ABC

import numpy as np
import scipy
import tensorflow
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, TextVectorization
from tensorflow.keras.layers import LSTM as LSTMLayer
from tensorflow.keras import Input
from keras.layers import Embedding, GlobalMaxPooling1D
from sklearn.feature_extraction.text import TfidfVectorizer
from data.GloVeEmbedding import GloVeEmbedding

from .Model import Model


# print(tf.config.list_physical_devices('GPU'))

class LSTM(Model, ABC):
    def __init__(self, dataset, embedding):
        super().__init__(dataset)
        self.__train = dataset.train
        self.__vocab_size = dataset.vocab_size
        self.__embedding = embedding

    def train(self):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.__train["description"])
        train_seq = tokenizer.texts_to_sequences(self.__train["description"])
        train_seq = pad_sequences(train_seq, maxlen=100)
        vocab_size = len(tokenizer.word_index) + 1
        embedding_matrix = self.__embedding.compute_embedding_matrix(tokenizer, vocab_size)

        model = Sequential()
        model.add(Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=100, trainable=False))
        model.add(LSTMLayer(128, return_sequences=True, dropout=0.3))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(345, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["acc"])
        print(model.summary())

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        model.fit(np.array(train_seq), np.array(self.__train["label"]), batch_size=64,
                  validation_split=0.2, epochs=5, verbose=1, callbacks=[es])

    def evaluate(self, dataset):
        pass

# self.dataset = dataset
# self.tokenizer = Tokenizer(num_words=self.dataset.vocab_size)
# self.tokenizer.fit_on_texts(self.dataset.train["description"])
# sequences = self.tokenizer.texts_to_sequences(self.dataset.train["description"])
# self.data = pad_sequences(sequences, maxlen=100)

# if self.__use_tfidf:
#     vectorize = self.__process_to_tfidf()
#     self.__train = tensorflow.data.Dataset.from_tensor_slices((self.__train["description"],
#                                                                self.__train["label"]))
#     self.__train = self.__train.batch(64).map(lambda x, y: (vectorize(x), y))
#     model.add(Input(shape=(vectorize.vocabulary_size(),)))
#     model.add(vectorize)
# ectorize = TextVectorization(max_tokens=self.__vocab_size, output_mode="tf-idf")
# print(self.__train["description"].shape)
# vectorize.adapt(self.__train["description"])
# return vectorize

# transformed = self.__process_to_tfidf()
# model = Sequential()
# model.add(Embedding(transformed.shape[1], 100, input_length=transformed.shape[1]))
# model.add(LSTMLayer(100, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(345, activation='softmax'))
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(transformed, self.__train["label"], batch_size=32, validation_split=0.4, epochs=20)
