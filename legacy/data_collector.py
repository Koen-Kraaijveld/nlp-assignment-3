import os
import re
import openai

openai.key = os.getenv("OPENAI_API_KEY")

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": 'Give me 3 short and unique facts about an elephant. Do not include the word '
                                    '"elephant" in your response. Try to focus on bodily characteristics. Start each'
                                    'response with "This animal".'}
    ]
)

responses = completion.choices[0].message.content


def process_responses(responses):
    responses = responses.split("\n")
    for i in range(len(responses)):
        responses[i] = responses[i].split(" ")[1:]
        responses[i] = " ".join(responses[i])

    return responses


print(process_responses(responses))