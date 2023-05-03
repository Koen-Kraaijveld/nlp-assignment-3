import itertools
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

    def start_prompts(self):
        df = pd.DataFrame({"description": [], "label": []})
        categories = self.get_categories(self.args["categories_file"])
        for category in categories:
            responses = []
            variables = [self.args["length"], self.args["detail"], self.args["complexity"], self.args["prefix"]]
            variations = list(itertools.product(*variables))
            for i in tqdm(range(len(variations)), desc=f"Prompting ({category})"):
                variation = variations[i]
                prompt = self.prepare_prompt(category, length=variation[0], detail=variation[1],
                                             complexity=variation[2], prefix=variation[3])
                # print(f"\n{prompt}")
                response = self.make_safe_prompt(prompt)
                response = self.__clean_responses(response.split("\n"))
                # print(f"\n{response}")
                responses.append(response)

            for row in responses:
                row = pd.DataFrame({"description": row, "label": [category] * len(row)})
                df = pd.concat([df, row], ignore_index=True)
            df.to_csv("./data/recent/descriptions.csv", index=False)

    def prepare_prompt(self, entity, length=1, detail=None, complexity=None, prefix=None):
        template = self.args["prompt_template"]
        article = "an" if entity[0] in ["a", "e", "i", "o", "u"] else "a"
        detail = "very short and " if detail == "short" else "very detailed and " if detail == "long" else ""
        template = re.sub(f"<var1>", str(length), template)
        template = re.sub(f"<var2>", detail, template)
        template = re.sub(f"<var3>", f"{article} {entity}", template)
        template = re.sub(f"<var4>", f"{entity}", template)
        template = re.sub(f"<var5>", complexity, template)
        template = re.sub(f"<var6>", prefix.capitalize(), template)
        return template

    def __clean_responses(self, responses):
        cleaned = []
        for i in range(len(responses)):
            if responses[i] != "":
                # responses[i] = responses[i].split(" ")
                # responses[i] = " ".join(responses[i])
                cleaned.append(responses[i])
        return cleaned

    def make_safe_prompt(self, prompt, max_retries=30, timeout=20):
        try:
            return self.__make_prompt(prompt)
        except (openai.error.APIError, openai.error.RateLimitError):
            retries = 1
            while retries <= max_retries:
                print(f"f{retries}. Error. Restarting. Prompt = {prompt}")
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
