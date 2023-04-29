import os
import re

import openai


class Prompter:
    def __init__(self, api_key, args):
        self.api_key = api_key
        openai.key = api_key
        self.args = args

    def prepare_prompt(self, entity, length=1, detail=None):
        template = self.args["prompt_template"]
        article = "an" if entity[0] in ["a", "e", "i", "o", "u"] else "a"
        detail = "very short and " if detail == "short" else "very detailed and " if detail == "long" else ""
        template = re.sub(f"<var1>", str(length), template)
        template = re.sub(f"<var2>", detail, template)
        template = re.sub(f"<var3>", f"{article} {entity}", template)
        template = re.sub(f"<var4>", f"{entity}", template)
        return template

    def make_prompt(self, prompt):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user",
                 "content": prompt}
            ]
        )

        return completion.choices[0].message.content



