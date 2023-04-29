import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from train_test_data import split_data, clean_text

x_train, x_test, y_train, y_test = split_data(test_split=0.4)

tfidf_vec = TfidfVectorizer()
x_train = tfidf_vec.fit_transform(x_train)
x_test = tfidf_vec.transform(x_test)

model = LogisticRegression().fit(x_train, y_train)
score = model.score(x_test, y_test)


print(score)


def predict(prompt):
    prompt = clean_text(pd.Series(prompt), remove_stopwords=False)
    prompt = tfidf_vec.transform(prompt)
    print(f"{model.predict(prompt)}, {model.predict_proba(prompt)}")


predict(["This mammal is a rodent."])