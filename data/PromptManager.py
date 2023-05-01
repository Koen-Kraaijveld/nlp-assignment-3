import os
import re
import time

import openai
import pandas as pd
from tqdm import tqdm


class PromptManager:
    def __init__(self, api_key, args):
        self.api_key = api_key
        openai.key = api_key
        self.args = args

    def start_prompts(self, length=20):
        df = pd.DataFrame({"description": [], "label": []})
        categories = self.get_categories("./data/saved/categories.txt")
        for category in tqdm(categories, desc="Prompting"):
            prompt1 = self.prepare_prompt(category, length=length)
            response1 = self.__clean_responses(self.make_safe_prompt(prompt1).split("\n"))
            prompt2 = self.prepare_prompt(category, length=length, detail="short")
            response2 = self.__clean_responses(self.make_safe_prompt(prompt2).split("\n"))
            prompt3 = self.prepare_prompt(category, length=length, detail="long")
            response3 = self.__clean_responses(self.make_safe_prompt(prompt3).split("\n"))

            for row in [response1, response2, response3]:
                row = pd.DataFrame({"description": row, "label": [category] * len(row)})
                df = pd.concat([df, row], ignore_index=True)
            df.to_csv("./data/saved/descriptions.csv", index=False)

    def prepare_prompt(self, entity, length=1, detail=None):
        template = self.args["prompt_template"]
        article = "an" if entity[0] in ["a", "e", "i", "o", "u"] else "a"
        detail = "very short and " if detail == "short" else "very detailed and " if detail == "long" else ""
        template = re.sub(f"<var1>", str(length), template)
        template = re.sub(f"<var2>", detail, template)
        template = re.sub(f"<var3>", f"{article} {entity}", template)
        template = re.sub(f"<var4>", f"{entity}", template)
        return template

    def __clean_responses(self, responses):
        cleaned = []
        for i in range(len(responses)):
            if responses[i] != "":
                responses[i] = responses[i].split(" ")[1:]
                responses[i] = " ".join(responses[i])
                cleaned.append(responses[i])
        return cleaned

    def make_safe_prompt(self, prompt, max_retries=10, timeout=10):
        try:
            return self.__make_prompt(prompt)
        except openai.APIError:
            retries = 1
            print(f"f{retries}. Caught APIError exception. Restarting. Prompt = {prompt}")
            while retries <= max_retries:
                try:
                    return self.__make_prompt(prompt)
                except openai.APIError:
                    time.sleep(timeout)
                    retries += 1

    def __make_prompt(self, prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": prompt}
            ]
        )

        return completion.choices[0].message.content

    def get_categories(self, file_path):
        with open(file_path, 'r') as file:
            return file.read().split("\n")
