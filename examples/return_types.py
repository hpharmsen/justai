"""" Example showing the use of GPT Easy with data returned in a fixed format using dataclasses"""
import pathlib
import json

from justai import Agent

def get_story():
    """ Read story.txt from the path of the current file """
    path = pathlib.Path(__file__).parent / 'story.txt'
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    agent = Agent('gpt-4-turbo-preview')
    prompt = "Read the following story and give me a list of the persons involved. " +\
             "Return json with keys name, profession and house number\n\n" + get_story()

    data = agent.chat(prompt, return_json=True)
    print(json.dumps(data, indent=4))
