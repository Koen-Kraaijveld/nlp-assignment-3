import os

from data.PromptManager import PromptManager

# Start all your responses with
args = {
    "prompt_template": 'Give me <var1> <var2>unique descriptions of <var3>. Do not include the word '
                       '"<var4>" or any of its variations in your response. Use <var5> language in your response.'
                       '<var6>',
    "length": [20],
    "detail": ["short", "", "long"],
    "complexity": ["very simple", "simple", "complex", "very complex"],
    "prefix": ["it", "this", "a", "the", "with", "in", "by", "on", "after", "before", ""],
    "temperature": [0.2, 0.6, 1.0],
    "categories_file": "./data/saved/categories_25.txt"
}


def start_prompts():
    """
    This function starts prompting ChatGPT with the OpenAI API key (stored as an environment variable) and the arguments
    shown above/
    """
    manager = PromptManager(os.getenv("OPENAI_API_KEY"), args)
    manager.start_prompts()


start_prompts()
