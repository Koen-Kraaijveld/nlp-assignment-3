import itertools
import os
import random


import pandas as pd
import numpy as np

from data.Dataset import TextClassificationDataset
from data.PromptManager import PromptManager
from models.LSTM import LSTM
# from legacy.word2vec import start
from data.GloVeEmbedding import GloVeEmbedding

args = {
    "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                       '"<var4>" or any of its variations in your response. Use <var5> language in your response. '
                       'Start all your responses with "<var6>".',
    "length": [20],
    "detail": ["short", "", "long"],
    "complexity": ["very simple", "simple", "complex", "very complex"],
    "prefix": ["it", "this", "a", "the"],
    "categories_file": "./data/saved/categories_100.txt"
}


def randomize_categories(save_file_path, read_file_path, num_elements=100):
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    categories = manager.get_categories(read_file_path)
    random.shuffle(categories)
    randomized = sorted(categories[:num_elements])
    with open(save_file_path, "w+") as file:
        for category in randomized:
            file.write(f"{category}\n")


def concatenate_dataframes(df1, df2):
    df = pd.concat([df1, df2], ignore_index=True)
    df.to_csv("./data/saved/raw_descriptions_100.csv", index=False)


def start_prompts():
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts()


# start_prompts()


glove = GloVeEmbedding("./data/embeddings/glove.6B.100d.txt")
dataset = TextClassificationDataset("data/saved/raw_descriptions_16.csv", test_split=0.4, shuffle=True)
model = LSTM(dataset, load_model_path="./models/saved/lstm.h5", save_tokenizer="./models/saved/tokenizer.json")
model.train(embedding=glove)

# text = ["This fruit is round and red."]
# print(model.predict(text))
