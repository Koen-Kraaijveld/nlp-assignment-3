import unittest

from data.GloVeEmbedding import GloVeEmbedding
from util import load_categories


class TestGloVeEmbedding(unittest.TestCase):
    def setUp(self) -> None:
        self.embeddings = GloVeEmbedding("../data/embeddings/glove.6B.100d.txt")

    def test_visualize_words_20(self):
        words = load_categories("../data/saved/categories_20.txt")
        self.embeddings.visualize_words(words)

    def test_visualize_words_100(self):
        words = load_categories("../data/saved/categories_100.txt")
        self.embeddings.visualize_words(words)

    def test_visualize_words_100_special_words(self):
        words = load_categories("../data/saved/categories_100.txt")
        special_words = ['map', 'train', 'cactus', 'peas', 'bulldozer', 'ear', 'mosquito', 'computer', 'calendar',
                         'basketball', 'castle', 'rain', 'moustache', 'bird', 'shoe', 'submarine', 'microwave',
                         'toothpaste', 'butterfly', 'diamond', 'crown', 'house', 'television', 'pliers', 'goatee']
        self.embeddings.visualize_words(words, special_words=special_words)

    def test_calculate_k_most_distant_words_1(self):
        words = load_categories("../data/saved/categories_100.txt")
        vectors = self.embeddings.calculate_k_most_distant_words(words, k=25, n=100000)
        print(next(iter(vectors)))
