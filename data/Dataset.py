import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TextClassificationDataset:
    def __init__(self, csv_path, test_split, val_split=0., shuffle=False):
        self.data = pd.read_csv(csv_path)
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.data["description"] = self.__clean_text(self.data["description"])
        self.data["label"] = LabelEncoder().fit_transform(self.data["label"])
        self.train = self.data[:int(len(self.data) * (1 - test_split))]
        self.test = self.data[len(self.train):]
        if val_split > 0:
            temp = self.train
            self.train = self.train[:int(len(self.train) * (1 - val_split))]
            self.val = temp[len(self.train):]
        self.vocab_size = self.__count_vocab_size()

    def __clean_text(self, text):
        clean_text = []
        for sentence in text:
            sentence = sentence.lower()
            sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'\<a href', ' ', sentence)
            sentence = re.sub(r'&amp;', '', sentence)
            sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
            sentence = re.sub(r'<br />', ' ', sentence)
            sentence = re.sub(r'\'', ' ', sentence)
            clean_text.append(sentence)
        return pd.Series(clean_text)

    def __count_vocab_size(self):
        unique_words = []
        for row in self.data["description"]:
            for word in row.split(" "):
                if word not in unique_words and word != " " and word != "":
                    unique_words.append(word)
        return len(unique_words)

    def get_all_words(self):
        words = []
        for row in self.data["description"]:
            for word in row.split(" "):
                if word != " " and word != "":
                    words.append(word)
        return words

    def __str__(self):
        return str(self.data)


