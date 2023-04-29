import os

from data.PromptManager import PromptManager
from data.Prompter import Prompter


args = {
    "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                       '"<var4>" or any of its variations in your response. Be as diverse as possible when '
                       'starting a new response.'
}


def start_prompts():
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts(length=1)


start_prompts()
