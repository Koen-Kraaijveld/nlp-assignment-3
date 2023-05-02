import os

from data.Dataset import TextClassificationDataset
from data.PromptManager import PromptManager
from models.LSTM import LSTM


args = {
    "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                       '"<var4>" or any of its variations in your response. Be as diverse as possible when '
                       'starting a new response.'
}


def start_prompts():
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts(length=20)


dataset = TextClassificationDataset("data/saved/raw_descriptions.csv", test_split=0.4, shuffle=True)

model = LSTM(dataset, use_tfidf=True)
model.train()
