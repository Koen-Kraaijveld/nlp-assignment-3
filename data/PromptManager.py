import os

import pandas as pd

from .Prompter import Prompter


class PromptManager:
    def __init__(self, api_key, args):
        self.prompter = Prompter(api_key, args)

    def start_prompts(self, length=20):
        df = pd.DataFrame({"description": [], "label": []})
        categories = self.get_categories("./data/saved/categories.txt")
        for category in categories[:3]:
            prompt1 = self.prompter.prepare_prompt(category, length=length)
            response1 = self.prompter.make_prompt(prompt1)
            prompt2 = self.prompter.prepare_prompt(category, length=length, detail="short")
            response2 = self.prompter.make_prompt(prompt2)
            prompt3 = self.prompter.prepare_prompt(category, length=length, detail="long")
            response3 = self.prompter.make_prompt(prompt3)

            for row in [response1, response2, response3]:
                row = {"description": row, "label": category}
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv("./data/saved/descriptions.csv", index=False)

    def get_categories(self, file_path):
        with open(file_path, 'r') as file:
            return file.read().split("\n")
