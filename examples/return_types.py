"""" Example showing the use of Justai with data returned in a json format """
import pathlib
import json


from justai import Model


def get_story():
    """ Read story.txt from the path of the current file """
    path = pathlib.Path(__file__).parent / 'story.txt'
    with open(path) as f:
        return '<story>\n' + f.read() + '\n</story>\n'


def json_example():
    model = Model('gemini-1.5-flash')
    prompt = "Read the following story and give me a list of the persons involved. " + \
             "Return json with keys name, profession and house number\n\n" + get_story()

    data = model.chat(prompt, return_json=True)
    print(json.dumps(data, indent=4))
    print(model.last_token_count())  # (input_token_count, output_token_count, total_token_count)


def structured_output_with_type_annotations():
    """ As of now, only the Google models support structured output with Python type annotations"""
    from typing_extensions import TypedDict  # Important: Use from typing_extensions instead of typing

    class Person(TypedDict):
        name: str
        house_number: int
        profession: str        
    persons = list[Person]

    model = Model('gemini-1.5-pro')
    prompt = "Read the following story and give me a list of the persons involved.\n\n" + get_story()
    data = model.chat(prompt, response_format=persons, cached=False)
    for person in data:
        print(person)


def structured_output_with_pydantic():
    """ As of now, only the OpenAI models support structured output with Pydantic"""
    from pydantic import BaseModel
    
    class Person(BaseModel):
        name: str
        house_number: int
        profession: str

    class Persons(BaseModel):
        persons: list[Person]
   
    model = Model('gpt-4o-2024-08-06')
    prompt = "Read the following story and give me a list of the persons involved.\n\n" + get_story()
    
    data = model.chat(prompt, response_format=Persons, cached=False)
    for person in data.persons:
        print(person)


if __name__ == "__main__":
    json_example()
    structured_output_with_type_annotations()
    structured_output_with_pydantic()
    