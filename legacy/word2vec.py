import gensim

def start(dataset):
    sentences = dataset.get_all_words()
    print(sentences)
    w2v = gensim.models.Word2Vec(sentences)

    print(w2v.wv.most_similar("creature", topn=3))