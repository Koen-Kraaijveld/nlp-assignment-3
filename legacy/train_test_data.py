import re

import nltk
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/saved/raw_descriptions.csv")


def clean_text(x, remove_stopwords=False):
    clean_x = []
    for text in x:
        text = text.lower()

        text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'\<a href', ' ', text)
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'\'', ' ', text)

        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        # text = nltk.WordPunctTokenizer().tokenize(text)
        # lemm = nltk.stem.WordNetLemmatizer()
        # text = list(map(lambda word: list(map(lemm.lemmatize, word)), text))
        clean_x.append(text)
    return pd.Series(clean_x)


x = df["description"]
y = df["label"]
x = clean_text(x, remove_stopwords=False)

def split_data(test_split):
    return train_test_split(x, y, test_size=test_split)
