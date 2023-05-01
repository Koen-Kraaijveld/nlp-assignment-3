import os

import pandas

from data.PromptManager import PromptManager


args = {
    "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                       '"<var4>" or any of its variations in your response. Be as diverse as possible when '
                       'starting a new response.'
}


def start_prompts():
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts(length=20)


# start_prompts()

# print("asdf")

# response = ["1. abc", "2. 123", "3. def"]
# print(pandas.Series(response))
#
# labels = ["airplane"] * len(response)
# print(pandas.Series(labels))
#
# df = pandas.DataFrame({"description": response, "label": labels})
# print(df)
