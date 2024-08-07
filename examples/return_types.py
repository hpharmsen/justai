"""" Example showing the use of Justai with data returned in a json format """
import pathlib
import json

from pydantic import BaseModel
from justai import Agent


def get_story():
    """ Read story.txt from the path of the current file """
    path = pathlib.Path(__file__).parent / 'story.txt'
    with open(path) as f:
        return f.read()


def json_example():
    agent = Agent('claude-3-5-sonnet-20240620')
    prompt = "Read the following story and give me a list of the persons involved. " + \
             "Return json with keys name, profession and house number\n\n" + get_story()

    data = agent.chat(prompt, return_json=True)
    print(json.dumps(data, indent=4))
    print(agent.last_token_count())  # (input_token_count, output_token_count, total_token_count)


def structured_output_example():
    """ As of now, only some OpenAI models support structured output"""

    class Person(BaseModel):
        name: str
        profession: str
        house_number: int

    class Persons(BaseModel):
        persons: list[Person]

    agent = Agent('gpt-4o-2024-08-06')
    prompt = "Read the following story and give me a list of the persons involved.\n\n" + get_story()
    data = agent.chat(prompt, response_format=Persons, cached=False)
    for person in data.persons:
        print(person)


if __name__ == "__main__":
    json_example()
    structured_output_example()