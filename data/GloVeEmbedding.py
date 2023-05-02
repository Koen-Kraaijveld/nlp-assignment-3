import numpy as np


class GloVeEmbedding:
    def __init__(self, file_path):
        self.embedding_index = self.__get_embedding_index(file_path)
        # self.embedding_matrix = self.__compute_embedding_matrix()

    def __get_embedding_index(self, file_path):
        embeddings_index = dict()
        file = open(file_path, encoding="utf8")
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
        file.close()
        return embeddings_index

    def compute_embedding_matrix(self, tokenizer, size_of_vocabulary):
        embedding_matrix = np.zeros((size_of_vocabulary, 100))

        for word, i in tokenizer.word_index.items():
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix
