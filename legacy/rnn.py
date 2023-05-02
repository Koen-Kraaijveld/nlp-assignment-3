import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from keras.layers import Embedding

from data.Dataset import dataset


vocab_size = dataset.__vocab_size
tokenizer = Tokenizer(num_words=vocab_size)

print(len(dataset.data))
tokenizer.fit_on_texts(dataset.data["description"])

sequences = tokenizer.texts_to_sequences(dataset.data["description"])
data = pad_sequences(sequences, maxlen=50)

model = Sequential()
model.add(Embedding(vocab_size, 100, input_length=50))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(data, np.array(dataset.data["animal"]), batch_size=32, validation_split=0.4, epochs=20)

new_animal = tokenizer.texts_to_sequences(["This animal is a rodent."])
new_animal = pad_sequences(new_animal, maxlen=50)
print(model.predict(new_animal))
