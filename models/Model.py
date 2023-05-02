from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, dataset):
        self.dataset = dataset

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self, dataset):
        pass