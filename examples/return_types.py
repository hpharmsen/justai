"""" Example showing the use of Justai with data returned in a json format """
import pathlib
import json


from justai import Model


def get_story():
    """ Read story.txt from the path of the current file """
    path = pathlib.Path(__file__).parent / 'story.txt'
    with open(path) as f:
        return '<story>\n' + f.read() + '\n</story>\n'


def json_example(model_name: str):
    model = Model(model_name)
    prompt = "Read the following story and give me a list of the persons involved. " + \
             "Return json with keys name, profession and house number\n\n" + get_story()
    return model.prompt(prompt, return_json=True, cached=False)


def structured_output_with_type_annotations(model_name: str):
    """ As of now, only the Google models support structured output with Python type annotations"""
    from typing_extensions import TypedDict  # Important: Use from typing_extensions instead of typing

    class Person(TypedDict):
        name: str
        house_number: int
        profession: str        
    persons = list[Person]

    model = Model(model_name)
    prompt = "Read the following story and give me a list of the persons involved.\n\n" + get_story()
    return model.prompt(prompt, response_format=persons, cached=False)


def structured_output_with_pydantic(model_name: str):
    """ As of now, only the OpenAI models support structured output with Pydantic"""
    from pydantic import BaseModel
    
    class Person(BaseModel):
        name: str
        house_number: int
        profession: str

    class Persons(BaseModel):
        persons: list[Person]
   
    model = Model(model_name)
    prompt = "Read the following story and give me a list of the persons involved.\n\n" + get_story()
    
    return model.prompt(prompt, response_format=Persons, cached=False)



if __name__ == "__main__":
    data = json_example('gemini-1.5-flash')
    print(json.dumps(data, indent=4))

    data = structured_output_with_type_annotations('gemini-1.5-pro')
    for person in data:
        print(person)

    data = structured_output_with_pydantic('gpt-5')
    for person in data.persons:
        print(person)
    