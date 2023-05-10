import re

from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Dataset:
    """
    Class to handle all necessary functionality and data handling related to the training, test and validation data.
    """
    def __init__(self, csv_path, test_split, val_split=0., shuffle=False, remove_stopwords=False):
        """
        Constructor for the Dataset class.
        :param csv_path: Path to the CSV file that contains all the data.
        :param test_split: Fraction of the full data to split into the test data.
        :param val_split: Fraction of the training data to be split into the validation data.
        :param shuffle: Boolean to determine if the full data should be shuffled before splitting.
        :param remove_stopwords: Boolean to determine if stopwords (e.g., 'a', 'the', etc.) should be removed during
        preprocessing.
        """
        self.data = pd.read_csv(csv_path)
        self.data["description"] = self.clean_text(self.data["description"], remove_stopwords=remove_stopwords)
        self.__label_encoder = LabelEncoder()
        self.data["label"] = self.__label_encoder.fit_transform(self.data["label"])
        if shuffle:
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.train = self.data[:int(len(self.data) * (1 - test_split))]
        self.test = self.data[len(self.train):]
        if val_split > 0:
            temp = self.train
            self.train = self.train[:int(len(self.train) * (1 - val_split))]
            self.val = temp[len(self.train):]
        self.vocab_size = self.__count_vocab_size()
        np.save("./models/saved/labels.npy", self.__label_encoder.classes_)

    def clean_text(self, text, remove_stopwords=False):
        """
        Class function that cleans all the data by removing punctuation, hyperlinks, double whitespaces, etc.
        :param text: Raw input text split into sentences.
        :param remove_stopwords: Boolean to determine if stopwords should be removed during cleaning.
        :return: Returns a Pandas Series that contains the cleaned text.
        """
        clean_text = []
        for sentence in text:
            sentence = sentence.lower()
            sentence = re.sub(r'https?:\/\/.*[\r\n]*', '', sentence, flags=re.MULTILINE)
            sentence = re.sub(r'\<a href', ' ', sentence)
            sentence = re.sub(r'&amp;', '', sentence)
            sentence = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', sentence)
            sentence = re.sub(r'<br />', ' ', sentence)
            sentence = re.sub(r'\'', ' ', sentence)
            sentence = re.sub(r'^(\d{1,2})(.|\)) ', '', sentence)
            sentence = re.sub(r'  ', ' ', sentence)

            if remove_stopwords:
                sentence = sentence.split()
                stops = set(stopwords.words("english"))
                sentence = [w for w in sentence if not w in stops]
                sentence = " ".join(sentence)

            clean_text.append(sentence)
        return pd.Series(clean_text)

    def __count_vocab_size(self):
        """
        Private class function that counts the number of unique words (vocabulary) in the dataset.
        :return: Vocabulary size
        """
        unique_words = []
        for row in self.data["description"]:
            for word in row.split(" "):
                if word not in unique_words and word != " " and word != "":
                    unique_words.append(word)
        return len(unique_words)

    def get_all_words(self):
        """
        Class function to get all the words that are not whitespaces or empty spaces in the full data.
        :return: Returns all the words that are not whitespaces or empty spaces in the full data.
        """
        words = []
        for row in self.data["description"]:
            for word in row.split(" "):
                if word != " " and word != "":
                    words.append(word)
        return words

    def decode_label(self, label):
        """
        Decodes an encoded label back to its original string using the LabelEncoder class
        :param label: Array of strings that contains the encoded labels as integers.
        :return: Returns an array of strings that contains the decoded labels as strings.
        """
        return self.__label_encoder.inverse_transform(label)

    def __str__(self):
        """
        Class function used for printing.
        :return: String representation of the Pandas Dataframe containing the full data.
        """
        return str(self.data)


