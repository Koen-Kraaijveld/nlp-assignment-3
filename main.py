import itertools
import os
import random

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


def start_prompts():
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts()


start_prompts()

# randomize_categories(save_file_path="./data/saved/categories_20.txt",
#                      read_file_path="./data/saved/categories_100.txt",
#                      num_elements=20)

# manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
# response = manager.make_safe_prompt(
#     'Give me 20 very detailed and unique descriptions of an angel. Do not include the word '
#     '"angel" or any of its variations in your response. Use very complex language in your '
#     'response. Start all your responses with "This".')
# print(response)

# glove = GloVeEmbedding("./data/embeddings/glove.6B.100d.txt")
# dataset = TextClassificationDataset("data/saved/raw_descriptions_16.csv", test_split=0.4, shuffle=True)
# model = LSTM(dataset, embedding=glove)
# model.train()

# manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
# variables = [args["length"], args["detail"], args["complexity"], args["prefix"]]
# variations = list(itertools.product(*variables))
# for i in range(len(variations)):
#     variation = variations[i]
#     prompt = manager.prepare_prompt("airplane", length=variation[0], detail=variation[1],
#                                     complexity=variation[2], prefix=variation[3])
#     print(f"{i+1}) {prompt}")
